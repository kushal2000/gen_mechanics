"""Cached reward/tolerance curves for the coevolution_v2 arms.

A finished run's tensorboard never changes, so a cache keyed on (path, mtime)
is permanent; only the generation or link still running is re-read. Every arm
is a CHAIN of runs -- a co-evolution generation or a 24 h baseline link -- and
a chain's curve is its runs concatenated in order, so the x axis is cumulative
epochs across the whole arm.
"""
import glob, hashlib, json, pathlib, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path("/share/portal/kk837/gen_mechanics")
LOGS = ROOT / "debug_outputs/train_logs/17sep_coevolution"
CACHE = ROOT / "debug_outputs/17sep_coevo_analysis/.curve_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SEC_PER_EPOCH = 5.9
"""Measured over the whole experiment: 140.4 h of sacct elapsed for
coevolution_v2's 90,000 epochs, including its 18 scene builds."""


def _curves(events: str):
    ev = pathlib.Path(events)
    f = CACHE / (hashlib.md5(f"{ev}:{ev.stat().st_mtime_ns}".encode()).hexdigest() + ".npz")
    if f.exists():
        z = np.load(f); return z["reward"], z["tol"]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    a = EventAccumulator(str(ev), size_guidance={"scalars": 0}); a.Reload()
    r = np.array([x.value for x in a.Scalars("rewards/iter")])
    t = np.array([x.value for x in a.Scalars("current_success_tolerance")])
    n = min(len(r), len(t)); r, t = r[:n], t[:n]
    np.savez(f, reward=r, tol=t); return r, t


def scalar(job: str, tag: str) -> np.ndarray:
    """One named tensorboard scalar from one run, cached on (events, tag)."""
    ev = pathlib.Path(_events(job))
    f = CACHE / (hashlib.md5(f"{ev}:{ev.stat().st_mtime_ns}:{tag}".encode()).hexdigest() + ".npy")
    if f.exists():
        return np.load(f)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    a = EventAccumulator(str(ev), size_guidance={"scalars": 0}); a.Reload()
    v = np.array([x.value for x in a.Scalars(tag)]) if tag in a.Tags()["scalars"] else np.array([])
    np.save(f, v); return v


def generation_jobs(label: str) -> dict:
    """``{generation: job id}`` for a co-evolution arm."""
    P = ROOT / "assets/populations" / label
    out = {}
    for d in sorted(P.glob("gen_*"), key=lambda p: int(p.name.split("_")[1])):
        j = d / "job_id.txt"
        if j.exists():
            out[int(d.name.split("_")[1])] = j.read_text().strip()
    return out


def _events(job: str, root: pathlib.Path = LOGS):
    ev = sorted(glob.glob(f"{root}/*_{job}_*/rank_0/*/summaries/events*"))
    return ev[-1] if ev else None


def _concat(jobs):
    """``(reward, tolerance, bounds)`` over a chain; bounds are the cumulative
    epoch counts at which one run handed over to the next."""
    rs, ts, bounds = [], [], []
    for j in jobs:
        ev = _events(j)
        if not ev:
            continue
        r, t = _curves(ev)
        if not len(r):
            continue
        rs.append(r); ts.append(t); bounds.append(sum(len(x) for x in rs))
    return np.concatenate(rs), np.concatenate(ts), bounds


def coevo(label: str):
    """A co-evolution arm, in generation order, plus its selection summaries."""
    P = ROOT / "assets/populations" / label
    gens = sorted(int(p.name.split("_")[1]) for p in P.glob("gen_*"))
    jobs, sel = [], []
    for g in gens:
        jid = (P / f"gen_{g}/job_id.txt")
        if jid.exists():
            jobs.append(jid.read_text().strip())
        s = P / f"gen_{g}/selection.json"
        if s.exists():
            sel.append(json.load(open(s))["summary"])
    r, t, b = _concat(jobs)
    return r, t, b, sel


def baseline(label: str = "coevolution_v2_baseline"):
    """The fixed-population arm: its 24 h links in order, read off the run
    directory names (``..._c01_...``), so a new link needs no bookkeeping."""
    runs = sorted(glob.glob(f"{LOGS}/*_{label}_c[0-9][0-9]_*"),
                  key=lambda p: pathlib.Path(p).name.split("_c")[1][:2])
    jobs = [pathlib.Path(p).name.split("_s100_")[1].split("_")[0] for p in runs]
    return _concat(jobs)


def imitation_v1():
    """Capsule SHARPA trained alone (job 786519). NOT a v2 arm: it ran under
    the palm-centre observation and the random 1200-object pool, so it shares
    only the task and the model with the curves above."""
    return _curves(_events("786519", ROOT / "debug_outputs/train_logs/depth_d64"))


WARMUP = 200
"""Epochs blanked after each restart: rl_games refills its 100-game reward
average from the first, failed episodes of a fresh process, so every generation
and link opens with a spurious collapse."""


def blank_restarts(v, starts, w: int = WARMUP):
    v = np.asarray(v, float).copy()
    for i in starts:
        v[i:i + w] = np.nan
    return v


def bridge(v):
    """Interpolate across the blanked windows, so a curve draws continuously."""
    v = np.asarray(v, float); m = np.isfinite(v); i = np.arange(len(v))
    return np.interp(i, i[m], v[m]) if m.any() else v


def smooth(v, w: int = 50):
    m = np.isfinite(v); x = np.where(m, v, 0.0)
    c = np.cumsum(np.insert(x, 0, 0.0)); k = np.cumsum(np.insert(m, 0, 0))
    i = np.arange(len(v)); lo = np.maximum(0, i - w + 1)
    n = k[i + 1] - k[lo]; s = c[i + 1] - c[lo]
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def epoch_hour_ticks(ax, step: int = 20000, ink: str = "#52514e"):
    import matplotlib.ticker as mt
    ax.xaxis.set_major_locator(mt.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(
        lambda e, _: f"{e/1000:.0f}k\n{e * SEC_PER_EPOCH / 3600:.0f} h"))
    ax.set_xlabel("Training epochs / wall-clock hours")
