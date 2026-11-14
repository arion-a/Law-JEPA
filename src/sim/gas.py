"""2D hard-disk gas simulator (units: mass=1, k_B=1). Numba time-stepped, cell-list collisions.

Frames are uint8 images (32 rows x 64 cols): value 0..3 = particle count in the pixel (capped),
value 4 = piston (right wall) column. Model input mapping: count/3 in [0,1], piston -> 1.0.
"""
import numpy as np
from numba import njit

R = 0.003          # disk radius
DT = 0.0005        # simulation step
STEPS = 80         # steps per frame
FRAME_DT = STEPS * DT
PPU = 32           # pixels per unit length
IMG_H, IMG_W = 32, 64
CS = 0.04          # collision cell size
NY = int(1.0 / CS) + 1
NXMAX = int(2.0 / CS) + 2
PIST_FRAMES = 25
T_MIN, T_MAX = 0.3, 2.0


@njit(cache=True)
def advance(x, y, vx, vy, W, vp, wt, nsteps):
    """Advance nsteps. Right wall at x=W moves with velocity vp until it reaches wt.
    Returns (W, vp, impulse) where impulse = total momentum delivered to all walls."""
    n = x.shape[0]
    head = np.empty(NXMAX * NY, np.int64)
    nxt = np.empty(n, np.int64)
    imp = 0.0
    d2max = (2 * R) ** 2
    for _ in range(nsteps):
        if vp != 0.0:
            W += vp * DT
            if (vp > 0 and W >= wt) or (vp < 0 and W <= wt):
                W = wt
                vp = 0.0
        for i in range(n):
            x[i] += vx[i] * DT
            y[i] += vy[i] * DT
            if x[i] < 0.0:
                x[i] = -x[i]
                imp += 2.0 * abs(vx[i])
                vx[i] = abs(vx[i])
            elif x[i] > W:
                rel = vx[i] - vp
                if rel > 0.0:
                    imp += 2.0 * rel
                    vx[i] = vp - rel
                x[i] = W - (x[i] - W)
            if y[i] < 0.0:
                y[i] = -y[i]
                imp += 2.0 * abs(vy[i])
                vy[i] = abs(vy[i])
            elif y[i] > 1.0:
                imp += 2.0 * abs(vy[i])
                vy[i] = -abs(vy[i])
                y[i] = 2.0 - y[i]
            if x[i] < 0.0:
                x[i] = 0.0
            if x[i] > W:
                x[i] = W
            if y[i] < 0.0:
                y[i] = 0.0
            if y[i] > 1.0:
                y[i] = 1.0
        ncx = int(W / CS) + 1
        for c in range(ncx * NY):
            head[c] = -1
        for i in range(n):
            cx = min(int(x[i] / CS), ncx - 1)
            cy = min(int(y[i] / CS), NY - 1)
            c = cx * NY + cy
            nxt[i] = head[c]
            head[c] = i
        for cx in range(ncx):
            for cy in range(NY):
                i = head[cx * NY + cy]
                while i >= 0:
                    for k in range(5):
                        if k == 0:
                            j = nxt[i]
                        else:
                            if k == 1:
                                ox = 1
                                oy = 0
                            elif k == 2:
                                ox = -1
                                oy = 1
                            elif k == 3:
                                ox = 0
                                oy = 1
                            else:
                                ox = 1
                                oy = 1
                            qx = cx + ox
                            qy = cy + oy
                            if qx < 0 or qx >= ncx or qy >= NY:
                                continue
                            j = head[qx * NY + qy]
                        while j >= 0:
                            dx = x[j] - x[i]
                            dy = y[j] - y[i]
                            d2 = dx * dx + dy * dy
                            if d2 < d2max and d2 > 0.0:
                                dvn = (vx[i] - vx[j]) * dx + (vy[i] - vy[j]) * dy
                                if dvn > 0.0:  # approaching: exchange normal velocity components
                                    f = dvn / d2
                                    vx[i] -= f * dx
                                    vy[i] -= f * dy
                                    vx[j] += f * dx
                                    vy[j] += f * dy
                            j = nxt[j]
                    i = nxt[i]
    return W, vp, imp


def render(x, y, W):
    ix = np.minimum((x * PPU).astype(np.int64), IMG_W - 1)
    iy = np.minimum((y * PPU).astype(np.int64), IMG_H - 1)
    img = np.bincount(iy * IMG_W + ix, minlength=IMG_H * IMG_W).reshape(IMG_H, IMG_W)
    img = np.minimum(img, 3).astype(np.uint8)
    img[:, min(int(W * PPU), IMG_W - 1)] = 4
    return img


class Gas:
    def __init__(self, N, W, T, rng):
        self.N = N
        self.W = float(W)
        self.x = rng.uniform(0, W, N)
        self.y = rng.uniform(0, 1.0, N)
        v = rng.normal(0, np.sqrt(T), (2, N))
        v *= np.sqrt(T / (0.5 * np.mean(v[0] ** 2 + v[1] ** 2)))  # exact starting T
        self.vx, self.vy = v[0].copy(), v[1].copy()
        self.vp = 0.0
        self.wt = self.W

    def T(self):
        return 0.5 * float(np.mean(self.vx ** 2 + self.vy ** 2))

    def snapshot(self):
        return (self.x.copy(), self.y.copy(), self.vx.copy(), self.vy.copy(), self.W, self.vp, self.wt)

    def restore(self, s):
        self.x, self.y, self.vx, self.vy = (a.copy() for a in s[:4])
        self.W, self.vp, self.wt = s[4:]

    def frame(self, pre_gap=0, n_pre=1):
        """advance one frame (STEPS substeps). If pre_gap>0 also render n_pre pictures at pre_gap, 2*pre_gap, ... substeps
        BEFORE the frame into self.pre (oldest first). The trajectory itself is unchanged."""
        done, imp, pres = 0, 0.0, []
        if pre_gap:
            for j in range(n_pre, 0, -1):
                tgt = STEPS - pre_gap * j
                self.W, self.vp, i = advance(self.x, self.y, self.vx, self.vy, self.W, self.vp, self.wt, tgt - done)
                imp, done = imp + i, tgt
                pres.append(render(self.x, self.y, self.W))
            self.pre = np.stack(pres)
        self.W, self.vp, i = advance(self.x, self.y, self.vx, self.vy, self.W, self.vp, self.wt, STEPS - done)
        return imp + i


def make_episode(seed, nframes=300, N=None, W0=None, T0=None, n_events=None, pre_gap=0, n_pre=1):
    """Simulate one episode. Overrides N/W0/T0/n_events exist for tests only."""
    rng = np.random.default_rng(seed)
    N = int(rng.integers(128, 513)) if N is None else N
    W0 = rng.uniform(1.0, 2.0) if W0 is None else W0
    T0 = rng.uniform(0.5, 2.0) if T0 is None else T0
    k = int(rng.choice([0, 1, 2], p=[0.3, 0.35, 0.35])) if n_events is None else n_events
    times = []
    if k >= 1:
        times.append(int(rng.integers(50, 150 if k == 2 else 200)))
    if k == 2:
        times.append(int(rng.integers(times[0] + 60, 230)))
    g = Gas(N, W0, T0, rng)
    imgs, pres, Ws, Ts, imps, logf, events = [], [], [], [], [], [], []

    def run_frame(lf=0.0):
        imps.append(g.frame(pre_gap, n_pre))
        imgs.append(render(g.x, g.y, g.W))
        if pre_gap:
            pres.append(g.pre)
        Ws.append(g.W)
        Ts.append(g.T())
        logf.append(lf)

    f = 0
    while f < nframes:
        if times and f == times[0]:
            times.pop(0)
            kind = "heat" if rng.random() < 0.5 else "piston"
            done = False
            for _ in range(30):
                if kind == "heat":
                    fac = rng.uniform(0.7, 1.4)
                    if T_MIN <= g.T() * fac ** 2 <= T_MAX:
                        g.vx *= fac
                        g.vy *= fac
                        run_frame(float(np.log(fac)))
                        events.append((f, 0, float(fac)))
                        f += 1
                        done = True
                        break
                else:
                    wt = rng.uniform(1.0, 2.0)
                    if abs(wt - g.W) < 0.1 or f + PIST_FRAMES > nframes:
                        continue
                    if not (T_MIN * 1.05 <= g.T() * g.W / wt <= T_MAX * 0.97):
                        continue
                    snap, n0 = g.snapshot(), len(Ts)
                    g.wt = wt
                    g.vp = (wt - g.W) / (PIST_FRAMES * FRAME_DT)
                    for _ in range(PIST_FRAMES):
                        run_frame()
                    g.W, g.vp = wt, 0.0
                    if min(Ts[n0:]) >= T_MIN and max(Ts[n0:]) <= T_MAX:
                        events.append((f, 1, float(wt)))
                        f += PIST_FRAMES
                        done = True
                        break
                    g.restore(snap)
                    for lst in (imgs, pres, Ws, Ts, imps, logf):
                        del lst[n0:]
            if done:
                continue
        run_frame()
        f += 1

    Wa, Ta = np.array(Ws), np.array(Ts)
    imp = np.array(imps)
    perim = 2 * (Wa + 1.0)
    cs_i, cs_p = np.cumsum(imp), np.cumsum(perim * FRAME_DT)
    lag = np.arange(nframes) - 10
    base_i = np.where(lag >= 0, cs_i[np.maximum(lag, 0)], 0.0)
    base_p = np.where(lag >= 0, cs_p[np.maximum(lag, 0)], 0.0)
    P_meas = (cs_i - base_i) / (cs_p - base_p)          # smoothed over trailing 10 frames
    phi = N * np.pi * R ** 2 / Wa
    out = dict(
        frames=np.stack(imgs), N=np.int32(N), W=Wa.astype(np.float32), A=Wa.astype(np.float32),
        T=Ta.astype(np.float32), P_meas=P_meas.astype(np.float32),
        P_label=(N * Ta / Wa * (1 + 2 * phi)).astype(np.float32), phi=phi.astype(np.float32),
        logf=np.array(logf, np.float32), seed=np.int64(seed),
        events=np.array(events, np.float32).reshape(-1, 3),  # (frame, 0=heat/1=piston, factor/target width)
    )
    if pre_gap:
        out['pre'] = np.stack(pres)      # picture pre_gap substeps before each frame
    return out
