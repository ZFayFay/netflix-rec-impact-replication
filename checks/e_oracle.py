"""Check E: the 'oracle' test (true taste given as a fixed offset) done properly.
The notebook's version: 8 epochs of Adam, unpenalized per-title betas, unweighted mean -> 1.018 vs 1.096.
Here: exact MLE by full-batch L-BFGS (per-day backward, low memory), three variants.
Usage: python3 e_oracle.py [seed] -> results/e_oracle_w{seed}.json
"""

import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load
import torch

torch.set_num_threads(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
g = load(seed, upto="sim")
I, J, T = g["I"], g["J"], g["T_DAYS"]
HIST, RECS, CHOICE, WATCH_DAY, V_TRUE, BETA_TRUE = (
    g["HIST"],
    g["RECS"],
    g["CHOICE"],
    g["WATCH_DAY"],
    g["V_TRUE"],
    g["BETA_TRUE"],
)
V_T = torch.tensor(V_TRUE, dtype=torch.float64)
hist_b = torch.tensor(HIST)
wd_t = torch.tensor(WATCH_DAY)
recs_t = torch.tensor(RECS)
ch_t = torch.tensor(CHOICE)
zeros1 = torch.zeros(I, 1, dtype=torch.float64)

ii, tt = np.nonzero(CHOICE < J)
in_rec = (RECS[ii, tt] == CHOICE[ii, tt][:, None]).any(1)
recw = np.bincount(CHOICE[ii, tt][in_rec], minlength=J)
nimp = np.bincount(RECS.ravel(), minlength=J)


def nll_backward(beta_fn):
    """accumulate gradient day by day; returns total NLL as float"""
    tot = 0.0
    for t in range(T):
        beta_vec = beta_fn()
        Cm = torch.zeros(I, J, dtype=torch.float64).scatter_(1, recs_t[:, t], 1.0)
        u = (V_T + beta_vec * Cm).masked_fill(hist_b | (wd_t < t), -1e9)
        l = torch.nn.functional.cross_entropy(
            torch.cat([u, zeros1], 1), ch_t[:, t], reduction="sum"
        )
        l.backward()
        tot += l.item()
    return tot


def summarize(b, tag, O):
    O[tag] = {
        "mean": float(b.mean()),
        "impression_weighted_mean": float((b * nimp).sum() / nimp.sum()),
        "mean_over_titles_ge20_recommended_watches": float(b[recw >= 20].mean()),
        "min": float(b.min()),
        "sd": float(b.std()),
        "mean_exp": float(np.exp(b).mean()),
        "corr_with_true": float(np.corrcoef(b, BETA_TRUE)[0, 1]),
    }


O = {
    "seed": seed,
    "true": {
        "mean": float(BETA_TRUE.mean()),
        "impression_weighted_mean": float((BETA_TRUE * nimp).sum() / nimp.sum()),
        "mean_over_titles_ge20_recommended_watches": float(BETA_TRUE[recw >= 20].mean()),
        "sd": float(BETA_TRUE.std()),
        "mean_exp": float(np.exp(BETA_TRUE).mean()),
    },
    "n_titles_ge20_recommended_watches": int((recw >= 20).sum()),
    "recommended_watches_per_title_quartiles": [
        float(x) for x in np.percentile(recw, [0, 25, 50, 75, 100])
    ],
}

# (1) one common bonus
b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS([b0], max_iter=40, line_search_fn="strong_wolfe")


def c1():
    opt.zero_grad()
    return torch.tensor(nll_backward(lambda: b0.expand(J)))


t0 = time.time()
opt.step(c1)
O["common_scalar_beta_MLE"] = float(b0.item())
# numerical s.e. from finite-difference of the gradient
eps = 1e-3


def grad_at(x):
    b0.data[:] = x
    b0.grad = None
    nll_backward(lambda: b0.expand(J))
    return b0.grad.item()


x0 = O["common_scalar_beta_MLE"]
h = (grad_at(x0 + eps) - grad_at(x0 - eps)) / (2 * eps)
b0.data[:] = x0
O["common_scalar_beta_se"] = float(1 / np.sqrt(h))
print(json.dumps(O, indent=1), time.time() - t0, flush=True)

# (2) per-title betas with the SAME shrinkage the notebook's main model uses:
#     per-batch loss = mean CE + l2*mean(dev^2)*len(b)/n  ==> total objective = NLL + l2*bs*mean(dev^2), bs=4096, l2=1
lam = 4096.0 / J
beta0 = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
dev = torch.zeros(J, dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS(
    [beta0, dev], max_iter=120, line_search_fn="strong_wolfe", tolerance_grad=1e-5
)


def c2():
    opt.zero_grad()
    tot = nll_backward(lambda: beta0 + dev)
    pen = lam * (dev**2).sum()
    pen.backward()
    return torch.tensor(tot + pen.item())


t0 = time.time()
opt.step(c2)
summarize((beta0 + dev).detach().numpy(), "per_title_with_notebook_L2_shrinkage", O)
print(json.dumps(O, indent=1), time.time() - t0, flush=True)
json.dump(O, open(os.path.join(HERE, "results", f"e_oracle_w{seed}.json"), "w"), indent=1)

# (3) per-title betas, unpenalized (what the notebook's oracle cell targets), run to convergence
bj = (beta0 + dev).detach().clone().requires_grad_(True)
opt = torch.optim.LBFGS([bj], max_iter=150, line_search_fn="strong_wolfe", tolerance_grad=1e-5)


def c3():
    opt.zero_grad()
    return torch.tensor(nll_backward(lambda: bj))


t0 = time.time()
opt.step(c3)
summarize(bj.detach().numpy(), "per_title_unpenalized_MLE", O)
print(json.dumps(O, indent=1), time.time() - t0, flush=True)
json.dump(O, open(os.path.join(HERE, "results", f"e_oracle_w{seed}.json"), "w"), indent=1)
