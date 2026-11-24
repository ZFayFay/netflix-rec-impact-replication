"""Harness: runs the notebook's code cells verbatim
inside a namespace, with the global SEED overridable, so extra analyses reuse the exact code.

Usage:  ns = load(seed=42, upto='fit')   # 'data' | 'sim' | 'naive' | 'fit' | 'oracle' | 'div' | 'cf' | 'dec' | 'stress'
"""

import json, os, sys, io, contextlib
import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "rec_impact_replication.ipynb")
DATA = os.path.join(HERE, "data", "ratings.parquet")

# code-cell indices in the notebook
STAGES = [
    ("imports", 1),
    ("data", 3),
    ("sim", 5),
    ("naive", 7),
    ("fit", 9),
    ("fig1", 10),
    ("oracle", 12),
    ("div", 14),
    ("fig2", 15),
    ("cf", 17),
    ("fig3", 18),
    ("dec", 20),
    ("fig4", 21),
    ("stress", 23),
]


def cell_src(i):
    nb = json.load(open(NB))
    return "".join(nb["cells"][i]["source"])


def load(
    seed=42,
    upto="fit",
    skip=("fig1", "fig2", "fig3", "fig4"),
    patches=None,
    quiet=False,
    workdir=None,
    defs_only=(),
):
    """patches: dict stage -> list of (old, new) string replacements applied to that cell's source."""
    patches = patches or {}
    workdir = workdir or os.path.join(HERE, "work")
    os.makedirs(workdir, exist_ok=True)
    if not os.path.exists(os.path.join(workdir, "ratings.parquet")):
        os.symlink(DATA, os.path.join(workdir, "ratings.parquet"))
    os.chdir(workdir)
    ns = {"__name__": "__checks__"}
    for name, idx in STAGES:
        if name in skip:
            if name == upto:
                break
            continue
        src = cell_src(idx)
        if name == "imports":
            assert "SEED = 42" in src
            src = src.replace("SEED = 42", f"SEED = {seed}")
        if name in defs_only:
            cut = {
                "fit": "t0 = time.time()",
                "div": "rows = []",
                "cf": "truth_cf = evaluate",
                "dec": "rec_today = ",
            }[name]
            assert cut in src
            src = src.split(cut)[0]
        for old, new in patches.get(name, []):
            assert old in src, (name, old)
            src = src.replace(old, new)
        # last-expression display cells: harmless under exec
        buf = io.StringIO()
        if quiet:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"<cell {idx}:{name}>", "exec"), ns)
        else:
            print(f"--- [{name}] seed={seed}", flush=True)
            exec(compile(src, f"<cell {idx}:{name}>", "exec"), ns)
        ns.setdefault("_out", {})[name] = buf.getvalue()
        # free large intermediates the later cells never use (keeps memory low)
        import gc

        if name == "data":
            for k in ("df", "d", "u_", "uc"):
                if k in ns and k not in ("u_",):
                    ns.pop(k, None)
        if name == "naive":
            for k in ("rec_mask", "avail", "watched_now"):
                ns.pop(k, None)
        gc.collect()
        if name == upto:
            break
    return ns


import numpy as np


def fit_cached(ns, tag, recs=None, choice=None, wd=None, **kw):
    """Fit the notebook's choice model (the notebook's function, unmodified) and cache (BETA_HAT, V_HAT)."""
    import torch

    path = os.path.join(HERE, "cache", f"{tag}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        z = np.load(path)
        return z["beta"], z["v"]
    recs = ns["RECS"] if recs is None else recs
    choice = ns["CHOICE"] if choice is None else choice
    wd = ns["WATCH_DAY"] if wd is None else wd
    m = ns["fit_choice_model"](recs, choice, wd, **kw)
    with torch.no_grad():
        beta = m.beta.numpy().copy()
        v = m.base_util(torch.tensor(ns["HIST"], dtype=torch.float32)).numpy().copy()
    np.savez(path, beta=beta, v=v)
    return beta, v


def naive_ratio(g, recs, choice, wd):
    """Cell 7 of the notebook as a function (same operations)."""
    I, J, T = g["I"], g["J"], g["T_DAYS"]
    HIST = g["HIST"]
    rec_mask = np.zeros((I, T, J), bool)
    np.put_along_axis(rec_mask, recs, True, axis=2)
    avail = np.ones((I, T, J), bool)
    for t in range(T):
        avail[:, t] = ~(HIST | (wd < t))
    wn = np.zeros((I, T, J), bool)
    ii, tt = np.nonzero(choice < J)
    wn[ii, tt, choice[ii, tt]] = True
    r1 = wn[avail & rec_mask].mean()
    r0 = wn[avail & ~rec_mask].mean()
    return float(r1 / r0), float(r1), float(r0)


def day_sums(g, v, beta, recs, wd):
    """Probability-scale effects over the realized days, for utilities v (I,J) and bonuses beta (J,)."""
    I, J, T = g["I"], g["J"], g["T_DAYS"]
    HIST = g["HIST"]
    S = dict(T_p1=0.0, T_p0cf=0.0, nT=0, U_p0=0.0, U_p1cf=0.0, nU=0, eng=0.0, eng_norec=0.0)
    for t in range(T):
        watched = HIST | (wd < t)
        Cm = np.zeros((I, J), bool)
        np.put_along_axis(Cm, recs[:, t], True, axis=1)
        u0 = np.where(watched, -np.inf, v)
        e0 = np.exp(u0)
        e1 = np.exp(u0 + beta)
        Z = np.where(Cm, e1, e0).sum(1) + 1.0
        av = ~watched
        Tm = Cm & av
        Um = ~Cm & av
        S["T_p1"] += (e1 / Z[:, None])[Tm].sum()
        S["T_p0cf"] += (e0 / (Z[:, None] - e1 + e0))[Tm].sum()
        S["nT"] += int(Tm.sum())
        S["U_p0"] += (e0 / Z[:, None])[Um].sum()
        S["nU"] += int(Um.sum())
        S["eng"] += (1 - 1 / Z).sum()
        S["eng_norec"] += (1 - 1 / (e0.sum(1) + 1.0)).sum()
    return {
        "expected_naive_ratio": float((S["T_p1"] / S["nT"]) / (S["U_p0"] / S["nU"])),
        "ATT_ratio": float(S["T_p1"] / S["T_p0cf"]),
        "selection_ratio": float((S["T_p0cf"] / S["nT"]) / (S["U_p0"] / S["nU"])),
        "engagement": float(S["eng"] / (I * T)),
        "engagement_lift_vs_no_recs_pct": float(100 * (S["eng"] / S["eng_norec"] - 1)),
    }
