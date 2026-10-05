"""Cached reward/tolerance curves per run. Finished runs never change, so a cache keyed on (path, mtime) is permanent."""
import glob, hashlib, os, pathlib, warnings, numpy as np; warnings.filterwarnings("ignore")
CACHE = pathlib.Path("/share/portal/kk837/gen_mechanics/debug_outputs/10sep_coevo_analysis/.curve_cache"); CACHE.mkdir(parents=True, exist_ok=True)
def curves(events_path):
    ev = pathlib.Path(events_path); key = hashlib.md5(f"{ev}:{ev.stat().st_mtime_ns}".encode()).hexdigest()
    f = CACHE / f"{key}.npz"
    if f.exists():
        z = np.load(f); return z["reward"], z["tol"]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    a = EventAccumulator(str(ev), size_guidance={"scalars": 0}); a.Reload()
    r = np.array([x.value for x in a.Scalars("rewards/iter")]); t = np.array([x.value for x in a.Scalars("current_success_tolerance")])
    n = min(len(r), len(t)); r, t = r[:n], t[:n]
    np.savez(f, reward=r, tol=t); return r, t
def run_events(pattern):
    ev = sorted(glob.glob(pattern)); return ev[-1] if ev else None
def baseline():
    segs = []
    for j in ("783644", "947594", "188721", "305382", "349692"):   # c0 .. c4, chained via continue_run.sub
        ev = run_events(f"debug_outputs/train_logs/depth_d64/*_{j}_*/rank_0/*/summaries/events*")
        if ev: segs.append(curves(ev))
    return np.concatenate([s[0] for s in segs]), np.concatenate([s[1] for s in segs]), [sum(len(s[0]) for s in segs[:i+1]) for i in range(len(segs))]
def coevo(root="assets/populations/coevo_r500_v1"):
    R = pathlib.Path(root); rs, ts, bounds, sel = [], [], [], []
    for g in sorted(R.glob("gen_*"), key=lambda p: int(p.name.split("_")[1])):
        jid = (g/"job_id.txt").read_text().strip() if (g/"job_id.txt").exists() else None
        ev = run_events(f"debug_outputs/train_logs/coevo/*_{jid}_*/rank_0/*/summaries/events*") if jid else None
        if not ev: continue
        r, t = curves(ev); rs.append(r); ts.append(t); bounds.append(sum(len(x) for x in rs))
        if (g/"selection.json").exists():
            import json; sel.append(json.load(open(g/"selection.json"))["summary"])
    return np.concatenate(rs), np.concatenate(ts), bounds, sel

WARMUP = 200
"""Epochs blanked after every restart. rl_games' rewards/iter is the mean over its
last 100 completed games, and a fresh process refills that buffer from the first
episodes to finish -- the short, failed ones -- so every generation and every
baseline link opens with a spurious collapse to ~150 that recovers over 100-200
epochs (measured: median 120, worst 470). The window is fixed rather than fitted
so a real post-selection drop would still show past it."""
def blank_restarts(r, starts, w=WARMUP):
    r = r.astype(float).copy()
    for i in starts: r[i:i+w] = np.nan
    return r
def smooth(v, w=50):
    """Rolling mean that ignores NaN and stays NaN where the window is empty."""
    m = np.isfinite(v); x = np.where(m, v, 0.0)
    c = np.cumsum(np.insert(x, 0, 0.0)); k = np.cumsum(np.insert(m, 0, 0))
    i = np.arange(len(v)); lo = np.maximum(0, i - w + 1)
    n = k[i+1] - k[lo]; s = c[i+1] - c[lo]
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)

SEC_PER_EPOCH = 5.5
"""Measured: the baseline chain's 100.0 h of sacct elapsed over 65,520 epochs
(4 restarts). The coevolution chain came to 6.16 s/epoch over 79,526 epochs --
its 40 scene rebuilds and startups add ~15 h on top of the same training rate."""
def epoch_hour_ticks(ax, step=10000, ink="#52514e"):
    """Two-row x tick labels: epochs over wall-clock hours at SEC_PER_EPOCH."""
    import matplotlib.ticker as mt
    ax.xaxis.set_major_locator(mt.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda e, _: f"{e:,.0f}\n{e*SEC_PER_EPOCH/3600:.0f} h"))
    ax.set_xlabel("training epochs  /  wall-clock hours on 2 GPUs   (1 epoch = 24,576 envs × 16 steps ≈ 0.39 M env steps ≈ 5.5 s)\n"
                  "co-evolution's 40 scene rebuilds add ~15 h of wall clock not counted here", color=ink, fontsize=9.5)

def bridge(v):
    """Linear interpolation across NaN runs, so a blanked restart window draws as a
    straight segment between its neighbours instead of a gap."""
    v = np.asarray(v, float); m = np.isfinite(v); i = np.arange(len(v))
    return np.interp(i, i[m], v[m]) if m.any() else v

def imitation():
    """Capsule SHARPA: one hand written from the grammar to imitate SHARPA, trained alone (job 786519)."""
    ev = run_events("debug_outputs/train_logs/depth_d64/*_786519_*/rank_0/*/summaries/events*")
    return curves(ev)
