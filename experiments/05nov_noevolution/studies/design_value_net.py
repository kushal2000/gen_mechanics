"""A design value function: a per-finger transformer trained from scratch, one deterministic net (6 Nov).

    from design_value_net import fit_transformer
    predict = fit_transformer(designs, labels)          # -> predict(designs) -> np.ndarray

Tokens straight from the design (no URDF, ~0.1 ms per hand, hand_sampler.grammar.derive at the zero pose), one per
finger: its 5 joint slots, each [joint point in the palm frame (3, per 100 mm), signed axis in the palm frame (3), link
vector (3, per 100 mm), limits (2, rad), present (1)] -- 60 numbers -- plus its joint count. The same kind of
information as the policy's static joint tokens, but its own encoder (nothing shared with the policy). Architecture
like the policy's: d_model 128, 2 layers, 4 heads, feed-forward 512; masked mean + max pooling over the fingers, with
the hand's finger and motor counts, then an MLP to one number. Trained with MSE, AdamW 3e-4, batch 256, cosine
schedule (6 Nov: dropout 0.3, 15,000 steps at a constant lr; see fit_transformer).
"""
from __future__ import annotations

import numpy as np

SLOTS, MAX_FINGERS = 5, 6
TOK = SLOTS * 12 + 1


def finger_tokens(design: dict) -> tuple[np.ndarray, np.ndarray]:
    """(fingers x TOK token matrix, hand vector [fingers / 6, motors / 25])."""
    from hand_sampler.grammar import derive as dv
    from hand_sampler.grammar import hand as gh
    hand = gh.hand_from_dict(design)
    toks = []
    for i, f in enumerate(hand.fingers):
        g = dv.finger_geometry(f)
        B = np.asarray(g.B, dtype=float)                   # the finger's zero-pose frame in the palm frame
        x = B[:, 0]
        p = np.asarray(g.base_mm, dtype=float).copy()
        t = []
        for j in range(SLOTS):
            if j < len(f.joints):
                jt = f.joints[j]
                ax = B @ dv.joint_axis(hand, i, j)
                lo, hi = dv.joint_limits(hand, i, j)
                link = x * float(jt.length)
                t += [*(p / 100.0), *ax, *(link / 100.0), lo, hi, 1.0]
                p = p + link
            else:
                t += [0.0] * 12
        toks.append(t + [len(f.joints) / SLOTS])
    motors = sum(len(f.joints) for f in hand.fingers)
    return np.asarray(toks, np.float32), np.asarray([len(hand.fingers) / MAX_FINGERS, motors / 25.0], np.float32)


def batch(designs: list[dict]):
    import torch
    feats = [finger_tokens(d) for d in designs]
    F = max(len(t) for t, _ in feats)
    X = torch.zeros(len(feats), F, TOK)
    M = torch.zeros(len(feats), F, dtype=torch.bool)
    H = torch.stack([torch.from_numpy(h) for _, h in feats])
    for k, (t, _) in enumerate(feats):
        X[k, :len(t)] = torch.from_numpy(t)
        M[k, :len(t)] = True
    return X, M, H


def _net(d: int = 128, layers: int = 2, heads: int = 4, ff: int = 512, dropout: float = 0.0):
    import torch
    from torch import nn

    class FingerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(TOK, d)
            enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.Linear(2 * d + 2, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))

        def forward(self, X, M, H):
            e = self.enc(self.inp(X), src_key_padding_mask=~M)
            w = M.unsqueeze(-1).float()
            mean = (e * w).sum(1) / w.sum(1).clamp(min=1.0)
            mx = e.masked_fill(~M.unsqueeze(-1), -1e4).max(1).values
            return self.head(torch.cat([mean, mx, H], -1)).squeeze(-1)
    return FingerTransformer()


def fit_transformer(designs: list[dict], labels, steps: int = 15000, seed: int = 0, lr: float = 3e-4,
                    wd: float = 1e-4, dropout: float = 0.3, warmup: int = 200, device: str | None = None):
    """One net fitted to (designs, labels); returns predict(designs) -> np.ndarray. Defaults: the tuned settings of
    6 Nov (study_vf_tweaks.py: judged on the beam search's out-of-distribution designs) -- dropout 0.3, lr 3e-4,
    weight decay 1e-4, 15,000 steps at a constant lr after a linear warm-up (validation curves are flat; no early
    stopping needed)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X, M, H = (t.to(dev) for t in batch(designs))
    y = torch.as_tensor(np.asarray(labels, np.float32), device=dev)
    net = _net(dropout=dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        net.train()
        b = torch.randint(len(y), (256,), generator=g).to(dev)
        loss = ((net(X[b], M[b], H[b]) - y[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()

    def predict(ds: list[dict]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for k in range(0, len(ds), 2048):
                Xc, Mc, Hc = (t.to(dev) for t in batch(ds[k:k + 2048]))
                out.append(net(Xc, Mc, Hc).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    predict.net = net                                  # for saving (coevolve.py writes value_net.pt)
    return predict


def train_curve(designs, labels, val_designs, val_labels, val_groups=None, steps: int = 15000, lr: float = 3e-4,
                wd: float = 1e-4, eval_every: int = 500, warmup: int = 200, seed: int = 0, device: str | None = None):
    """Validation curve of one net: constant lr after a linear warm-up (so every point of the curve is a fair "stop
    here" estimate), evaluated every eval_every steps. Returns [{step, train_mse, val_mse, val_r, val_rank,
    val_rank_in_group}]."""
    import statistics as st
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X, M, H = (t.to(dev) for t in batch(designs))
    y = torch.as_tensor(np.asarray(labels, np.float32), device=dev)
    Xv, Mv, Hv = (t.to(dev) for t in batch(val_designs))
    yv = np.asarray(val_labels, np.float32)
    net = _net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rank(a):
        o = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
        return o

    def spear(a, b):
        return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    curve, run = [], []
    for k in range(1, steps + 1):
        net.train()
        b = torch.randint(len(y), (256,), generator=g).to(dev)
        loss = ((net(X[b], M[b], H[b]) - y[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(float(loss))
        if k % eval_every == 0:
            net.eval()
            with torch.no_grad():
                pv = net(Xv, Mv, Hv).cpu().numpy()
            row = {"step": k, "train_mse": float(np.mean(run[-eval_every:])), "val_mse": float(np.mean((pv - yv) ** 2)),
                   "val_r": float(np.corrcoef(pv, yv)[0, 1]), "val_rank": spear(pv, yv)}
            if val_groups is not None:
                gs = {}
                for i, gname in enumerate(val_groups):
                    gs.setdefault(gname, []).append(i)
                row["val_rank_in_group"] = float(np.nanmean([spear(pv[idx], yv[idx]) for idx in gs.values() if len(idx) > 4]))
            curve.append(row)
    return curve


def train_preds(designs, labels, evalsets: dict, steps: int = 15000, lr: float = 3e-4, wd: float = 1e-4,
                eval_every: int = 500, warmup: int = 200, seed: int = 0, bootstrap: bool = False,
                arch: dict | None = None, device: str | None = None):
    """Train one net (constant lr after a warm-up) and return {name: [(step, predictions)]} for every eval set
    (name -> designs) every eval_every steps -- for validation curves and early stopping."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X, M, H = (t.to(dev) for t in batch(designs))
    y = torch.as_tensor(np.asarray(labels, np.float32), device=dev)
    E = {k: tuple(t.to(dev) for t in batch(v)) for k, v in evalsets.items()}
    net = _net(**(arch or {})).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.randint(len(y), (len(y),), generator=g) if bootstrap else torch.arange(len(y))
    out = {k: [] for k in evalsets}
    for k in range(1, steps + 1):
        net.train()
        b = pool[torch.randint(len(pool), (256,), generator=g)].to(dev)
        loss = ((net(X[b], M[b], H[b]) - y[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if k % eval_every == 0:
            net.eval()
            with torch.no_grad():
                for name, (Xe, Me, He) in E.items():
                    pe = np.concatenate([net(Xe[i:i + 4096], Me[i:i + 4096], He[i:i + 4096]).cpu().numpy()
                                         for i in range(0, len(Xe), 4096)])
                    out[name].append((k, pe))
    return out
