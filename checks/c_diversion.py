"""Check C: how informative is the diversion validation (0.77 / 0.99)?
 - reproduce the notebook's numbers (same RNG call order)
 - sensitivity to experiment size (REPS) and experiment seed
 - placebo: a model with NO personalization at all (title intercepts + one common bonus)
Usage: python3 c_diversion.py   -> results/c_diversion.json
"""

import sys, os, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load, fit_cached
import torch

torch.set_num_threads(1)

g = load(42, upto="div", skip=("fig1", "fig2", "fig3", "fig4", "oracle"), defs_only=("fit", "div"))
I, J, T = g["I"], g["J"], g["T_DAYS"]
HIST, RECS, CHOICE, WATCH_DAY, WEND = (
    g["HIST"],
    g["RECS"],
    g["CHOICE"],
    g["WATCH_DAY"],
    g["WATCHED_END"],
)
V_TRUE, BETA_TRUE = g["V_TRUE"], g["BETA_TRUE"]
GENRES, gmask, arm_recs, choice_probs = g["GENRES"], g["gmask"], g["arm_recs"], g["choice_probs"]
beta_hat, v_hat = fit_cached(g, "base_w42_s0")
R = {}

# ---- placebo model: u_ij = delta_j + beta0*[rec]; no member information at all. Exact MLE by L-BFGS.
hist_b = torch.tensor(HIST)
wd_t = torch.tensor(WATCH_DAY)
recs_t = torch.tensor(RECS)
ch_t = torch.tensor(CHOICE)
delta = torch.full((J,), -6.0, dtype=torch.float64, requires_grad=True)
b0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS([delta, b0], max_iter=200, line_search_fn="strong_wolfe")


def clo():
    opt.zero_grad()
    tot = 0.0
    for t in range(T):
        Cm = torch.zeros(I, J, dtype=torch.float64).scatter_(1, recs_t[:, t], 1.0)
        u = (delta + b0 * Cm).masked_fill(hist_b | (wd_t < t), -1e9)
        tot = tot + torch.nn.functional.cross_entropy(
            torch.cat([u, torch.zeros(I, 1, dtype=torch.float64)], 1), ch_t[:, t], reduction="sum"
        )
    tot.backward()
    return tot


t0 = time.time()
opt.step(clo)
V_NULL = np.tile(delta.detach().numpy(), (I, 1))
B_NULL = np.full(J, b0.item())
R["placebo_no_personalization_beta0"] = float(b0.item())
R["placebo_multiplier_exp_beta0"] = float(np.exp(b0.item()))
print("placebo fit", time.time() - t0, R, flush=True)

MODELS = {"model": (v_hat, beta_hat), "truth": (V_TRUE, BETA_TRUE), "placebo": (V_NULL, B_NULL)}


def title_means(p, p0):
    return np.append(
        p.mean(0), p0.mean()
    )  # (J+1,) mean over members; group shares are linear in this


def shares(vec, g_from):
    out = {"g": vec[:J][gmask[g_from]].sum(), "nothing": vec[J]}
    for k in GENRES:
        if k != g_from:
            out[k] = vec[:J][gmask[k] & ~gmask[g_from]].sum()
    return out


def run(seed0, reps, checkpoints):
    """Same RNG call order as notebook cell 14: per (g, r): rng(seed0+r) -> rec_c, rec_a, sample ctl, sample arm."""
    acc = {
        g_: {
            k: np.zeros(J + 1)
            for k in ["exp_c", "exp_a"] + [m + s for m in MODELS for s in ("_c", "_a")]
        }
        for g_ in GENRES
    }
    res = {}
    for r in range(reps):
        for g_ in GENRES:
            rr = np.random.default_rng(seed0 + r)
            rec_c, rec_a = arm_recs(rr), arm_recs(rr, g_)
            for nm, rec in (("exp_c", rec_c), ("exp_a", rec_a)):
                p, p0 = choice_probs(V_TRUE, BETA_TRUE, rec, WEND)
                cum = np.hstack([p, p0[:, None]]).cumsum(1)
                ch = (rr.random((I, 1)) < cum).argmax(1)
                acc[g_][nm] += np.bincount(ch, minlength=J + 1) / I
            for m, (v, b) in MODELS.items():
                acc[g_][m + "_c"] += title_means(*choice_probs(v, b, rec_c, WEND))
                acc[g_][m + "_a"] += title_means(*choice_probs(v, b, rec_a, WEND))
        if (r + 1) in checkpoints:
            rows = []
            for g_ in GENRES:
                d = {}
                for m in ["exp"] + list(MODELS):
                    sa, sc = shares(acc[g_][m + "_a"], g_), shares(acc[g_][m + "_c"], g_)
                    lost = -(sa["g"] - sc["g"])
                    d[m] = {k: (sa[k] - sc[k]) / lost for k in sa if k != "g"}
                for k in d["exp"]:
                    rows.append(
                        dict(
                            blocked=g_,
                            to=k,
                            experiment=d["exp"][k],
                            model=d["model"][k],
                            truth=d["truth"][k],
                            placebo=d["placebo"][k],
                        )
                    )
            div = pd.DataFrame(rows)
            c = lambda a, b, df=div: float(np.corrcoef(df[a], df[b])[0, 1])
            ng = div[div.to != "nothing"]
            dm = div.copy()
            for col in ["experiment", "model", "truth", "placebo"]:
                dm[col] = dm[col] - dm.groupby("to")[col].transform(
                    "mean"
                )  # remove destination-size effect
            res[r + 1] = {
                "n_pairs": len(div),
                "model_vs_experiment": c("experiment", "model"),
                "model_vs_truth": c("truth", "model"),
                "placebo_vs_experiment": c("experiment", "placebo"),
                "placebo_vs_truth": c("truth", "placebo"),
                "truth_vs_experiment": c("experiment", "truth"),
                "excl_nothing_model_vs_truth": c("truth", "model", ng),
                "excl_nothing_placebo_vs_truth": c("truth", "placebo", ng),
                "excl_nothing_model_vs_experiment": c("experiment", "model", ng),
                "demeaned_by_destination_model_vs_truth": c("truth", "model", dm),
                "demeaned_by_destination_placebo_vs_truth": c("truth", "placebo", dm),
                "share_to_nothing_experiment": float(div[div.to == "nothing"].experiment.mean()),
                "share_to_nothing_truth": float(div[div.to == "nothing"].truth.mean()),
                "share_to_nothing_model": float(div[div.to == "nothing"].model.mean()),
                "share_to_nothing_placebo": float(div[div.to == "nothing"].placebo.mean()),
                "share_to_nothing_experiment_min_max": [
                    float(div[div.to == "nothing"].experiment.min()),
                    float(div[div.to == "nothing"].experiment.max()),
                ],
            }
            print(seed0, r + 1, json.dumps(res[r + 1]), flush=True)
            if seed0 == 500 and (r + 1) == reps:
                div.to_csv(
                    os.path.join(HERE, "results", "c_diversion_pairs_seed500.csv"), index=False
                )
    return res


R["seed500"] = run(500, 100, (5, 10, 25, 50, 100))
json.dump(R, open(os.path.join(HERE, "results", "c_diversion.json"), "w"), indent=1)
for s0 in (1500, 2500, 3500):
    R[f"seed{s0}"] = run(s0, 100, (5, 100))
    json.dump(R, open(os.path.join(HERE, "results", "c_diversion.json"), "w"), indent=1)
