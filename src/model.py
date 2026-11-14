"""Two-level JEPA (design doc section 6). Level 0: masked patch-code prediction of the next clip.
Level 1: predict the 8-number state s(t+50) from s(t) and the action."""
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

D, NP, SDIM = 128, 32, 8
GRID_R, GRID_C = 4, 8


def patchify(x):
    """(B,K,32,64) -> (B,32,K*64): 8x8 patches, all K frames of the clip stacked per patch"""
    B, K = x.shape[:2]
    return x.reshape(B, K, 4, 8, 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(B, NP, K * 64)


def block_masks():
    """All masks hiding two non-overlapping 2x2-patch blocks (8 of 32 patches = 25%)."""
    def block(r, c):
        return {(r + i) * GRID_C + (c + j) for i in range(2) for j in range(2)}
    anchors = [(r, c) for r in range(GRID_R - 1) for c in range(GRID_C - 1)]
    out = []
    for a, b in itertools.combinations(anchors, 2):
        s1, s2 = block(*a), block(*b)
        if not (s1 & s2):
            m = torch.zeros(NP, dtype=torch.bool)
            m[list(s1 | s2)] = True
            out.append(m)
    return torch.stack(out)


def encoder_stack(layers):
    layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(D), enable_nested_tensor=False)


class Backbone(nn.Module):
    """patch embedding + position tag + 4-layer transformer (E0)"""
    def __init__(self, nf=2):
        super().__init__()
        self.embed = nn.Linear(nf * 64, D)
        self.pos = nn.Parameter(torch.randn(NP, D) * 0.02)
        self.enc = encoder_stack(4)

    def forward(self, patches, vis_idx=None):
        tok = self.embed(patches) + self.pos
        if vis_idx is not None:
            tok = torch.gather(tok, 1, vis_idx[..., None].expand(-1, -1, D))
        return self.enc(tok)


class Pool(nn.Module):
    """G: average the patch codes, squeeze to s (8 numbers)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, SDIM))

    def forward(self, codes):
        return self.net(codes.mean(1))


class Predictor0(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(NP, D) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(D))
        self.enc = encoder_stack(2)
        self.out = nn.Linear(D, D)

    def forward(self, vis_codes, vis_idx):
        B = vis_codes.shape[0]
        tok = (self.mask_tok + self.pos).expand(B, NP, D).clone()
        tok = tok.scatter(1, vis_idx[..., None].expand(-1, -1, D), vis_codes + self.pos[vis_idx])
        return self.out(self.enc(tok))


class Predictor1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SDIM + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, SDIM))

    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], 1))


class HJEPA(nn.Module):
    def __init__(self, aux=False, aux_s=False, nf=2):
        super().__init__()
        self.aux_s = nn.Linear(SDIM, 2) if aux_s else None   # optional linear head: s -> global (log motion, log count)
        self.aux = nn.Linear(D, 2) if aux else None     # optional motion head (per-patch |frame diff|, count)
        self.nf = nf
        self.E0, self.G, self.P0, self.P1 = Backbone(nf), Pool(), Predictor0(), Predictor1()
        self.tE0, self.tG = copy.deepcopy(self.E0), copy.deepcopy(self.G)
        for p in itertools.chain(self.tE0.parameters(), self.tG.parameters()):
            p.requires_grad_(False)
        self.register_buffer("masks", block_masks())

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("tE0", "tG"))]

    @torch.no_grad()
    def ema_update(self, m):
        for pt, p in zip(itertools.chain(self.tE0.parameters(), self.tG.parameters()),
                         itertools.chain(self.E0.parameters(), self.G.parameters())):
            pt.mul_(m).add_(p.detach(), alpha=1 - m)

    def state(self, clip):
        """s(t) from the full, unmasked clip with the online encoder (used by probes)"""
        return self.G(self.E0(patchify(clip)))

    def losses(self, c_t, c_t1, c_t50, a, s_floor=0.5, aux_w=0.0, aux_s_w=0.0):
        B = c_t.shape[0]
        p_t, p_t1, p_t50 = patchify(c_t), patchify(c_t1), patchify(c_t50)
        # level 0: masked prediction of next clip's patch codes
        mask = self.masks[torch.randint(len(self.masks), (B,), device=c_t.device)]
        vis_idx = (~mask).nonzero()[:, 1].reshape(B, -1)
        codes = self.E0(p_t, vis_idx)
        pred = self.P0(codes, vis_idx)
        with torch.no_grad():
            tgt = F.layer_norm(self.tE0(p_t1), (D,))
        L0 = F.mse_loss(pred.float(), tgt.float())
        # level 1: predict state 50 frames ahead
        codes_full = self.E0(p_t)
        s = self.G(codes_full)
        s_pred = self.P1(s.float(), a)
        with torch.no_grad():
            s_tgt = self.tG(self.tE0(p_t50)).float()
        L1 = F.mse_loss(s_pred, s_tgt)
        # anti-collapse: VICReg variance/covariance on patch codes; total-spread floor on s
        z = codes.float().reshape(-1, D)
        std = torch.sqrt(z.var(0) + 1e-4)
        var = F.relu(1 - std).mean()
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
        spread = s.float().std(0).sum()
        sfl = F.relu(s_floor - spread)
        total = L0 + L1 + var + cov + sfl
        out = dict(L0=L0.item(), L1=L1.item(), var=var.item(), cov=cov.item(), s_spread=spread.item())
        if self.aux is not None and aux_w > 0:
            # self-supervised motion target computed from the clip's own pixels: per patch, sum|frame1-frame0| and sum count
            fr = p_t.float().reshape(B, NP, self.nf, 64)
            tgt = torch.stack([(fr[:, :, 1:] - fr[:, :, :-1]).abs().sum((2, 3)) / (self.nf - 1), fr[:, :, -1].sum(-1)], -1)
            La = F.mse_loss(self.aux(codes_full).float(), tgt)
            total = total + aux_w * La
            out["Laux"] = La.item()
        if self.aux_s is not None and aux_s_w > 0:
            fr = p_t.float().reshape(B, NP, self.nf, 64)
            g = torch.stack([torch.log((fr[:, :, 1:] - fr[:, :, :-1]).abs().sum((1, 2, 3)) / (self.nf - 1) + 1.0),
                             torch.log(fr[:, :, -1].sum((1, 2)) + 1.0)], 1)
            Ls = F.mse_loss(self.aux_s(s.float()), g)
            total = total + aux_s_w * Ls
            out["Laux_s"] = Ls.item()
        return total, out
