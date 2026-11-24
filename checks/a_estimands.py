"""Check A: what exactly are '7.5x', '3.1x', '3.4x'? Are they the same estimand?
Also: oracle test done properly, is V_TRUE a function of HIST, utility variance shares.
Usage: python3 a_estimands.py [world_seed]   -> writes results/a_estimands_w{seed}.json
"""

import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load, fit_cached, cell_src
import torch

torch.set_num_threads(1)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
do_oracle = (len(sys.argv) <= 2) or (sys.argv[2] != "nooracle")
g = load(seed, upto="fit", defs_only=("fit",))
I, J, T = g["I"], g["J"], g["T_DAYS"]
HIST, RECS, CHOICE, WATCH_DAY = g["HIST"], g["RECS"], g["CHOICE"], g["WATCH_DAY"]
V_TRUE, BETA_TRUE, pop = g["V_TRUE"], g["BETA_TRUE"], g["pop"]
R = {"seed": seed}

t0 = time.time()
beta_hat, v_hat = fit_cached(g, f"base_w{seed}_s0")
print(f"fit (or cache) {time.time()-t0:.0f}s  mean beta_hat {beta_hat.mean():.4f}", flush=True)

# ---------- 1. the three headline numbers, as the notebook computes them
R["naive_ratio_notebook"] = float(g["naive_ratio"])
R["rate_rec"] = float(g["rate_rec"])
R["rate_not"] = float(g["rate_not"])
R["true_mean_beta"] = float(BETA_TRUE.mean())
R["true_exp_of_mean_beta"] = float(np.exp(BETA_TRUE.mean()))
R["true_mean_exp_beta"] = float(np.exp(BETA_TRUE).mean())  # notebook's 'true_ratio'
R["true_sd_beta"] = float(BETA_TRUE.std())
R["model_mean_beta"] = float(beta_hat.mean())
R["model_exp_of_mean_beta"] = float(np.exp(beta_hat.mean()))
R["model_mean_exp_beta"] = float(np.exp(beta_hat).mean())  # notebook's 'model' multiplier
R["model_sd_beta"] = float(beta_hat.std())
R["corr_beta"] = float(np.corrcoef(BETA_TRUE, beta_hat)[0, 1])
R["slope_betahat_on_beta"] = float(np.polyfit(BETA_TRUE, beta_hat, 1)[0])


# ---------- 2. probability-scale, same-estimand comparison over the realized 40 days
def day_sums(v, beta):
    S = dict(T_p1=0.0, T_p0cf=0.0, nT=0, U_p0=0.0, U_p1cf=0.0, nU=0, eng=0.0, eng_norec=0.0)
    Y1 = np.zeros(J)
    Y0 = np.zeros(J)
    nav = np.zeros(J)
    nimp = np.zeros(J)
    for t in range(T):
        watched = HIST | (WATCH_DAY < t)
        Cm = np.zeros((I, J), bool)
        np.put_along_axis(Cm, RECS[:, t], True, axis=1)
        u0 = np.where(watched, -np.inf, v)
        e0 = np.exp(u0)
        e1 = np.exp(u0 + beta)
        Z = np.where(Cm, e1, e0).sum(1) + 1.0
        p1T = e1 / Z[:, None]  # shown (factual for treated cells)
        p0cf = e0 / (
            Z[:, None] - e1 + e0
        )  # treated cell with its bonus removed, rest of slate unchanged
        p0U = e0 / Z[:, None]  # not shown (factual for untreated cells)
        p1cf = e1 / (Z[:, None] - e0 + e1)  # untreated cell with a bonus added, rest unchanged
        av = ~watched
        Tm = Cm & av
        Um = ~Cm & av
        S["T_p1"] += p1T[Tm].sum()
        S["T_p0cf"] += p0cf[Tm].sum()
        S["nT"] += int(Tm.sum())
        S["U_p0"] += p0U[Um].sum()
        S["U_p1cf"] += p1cf[Um].sum()
        S["nU"] += int(Um.sum())
        S["eng"] += (1 - 1 / Z).sum()
        S["eng_norec"] += (1 - 1 / (e0.sum(1) + 1.0)).sum()
        Y1 += np.where(Tm, p1T, 0).sum(0) + np.where(Um, p1cf, 0).sum(0)
        Y0 += np.where(Tm, p0cf, 0).sum(0) + np.where(Um, p0U, 0).sum(0)
        nav += av.sum(0)
        nimp += Tm.sum(0)
    out = {}
    out["expected_naive_ratio"] = (S["T_p1"] / S["nT"]) / (S["U_p0"] / S["nU"])
    out["ATT_ratio_recommended_cells"] = (
        S["T_p1"] / S["T_p0cf"]
    )  # causal lift on the cells actually recommended
    out["selection_ratio"] = (S["T_p0cf"] / S["nT"]) / (
        S["U_p0"] / S["nU"]
    )  # how much more likely anyway
    out["ATU_ratio_not_recommended_cells"] = S["U_p1cf"] / S["U_p0"]
    out["ATE_ratio_all_cells_pooled"] = (S["T_p1"] + S["U_p1cf"]) / (S["T_p0cf"] + S["U_p0"])
    r = Y1 / Y0  # per-title Ybar_j(1)/Ybar_j(0)
    out["paper_style_ATEj_over_Y0_weighted_mean"] = float(((r - 1) * nimp).sum() / nimp.sum())
    out["paper_style_ATEj_over_Y0_mean"] = float((r - 1).mean())
    out["paper_style_ATEj_over_Y0_median"] = float(np.median(r - 1))
    out["engagement_current"] = S["eng"] / (I * T)
    out["engagement_no_recs_at_all"] = S["eng_norec"] / (I * T)
    out["engagement_lift_current_vs_no_recs_pct"] = 100 * (S["eng"] / S["eng_norec"] - 1)
    return {k: float(x) for k, x in out.items()}, nimp


R["truth_prob_scale"], nimp = day_sums(V_TRUE, BETA_TRUE)
R["model_prob_scale"], _ = day_sums(v_hat, beta_hat)
R["true_impression_weighted_mean_beta"] = float((BETA_TRUE * nimp).sum() / nimp.sum())
R["model_impression_weighted_mean_beta"] = float((beta_hat * nimp).sum() / nimp.sum())
R["true_impression_weighted_mean_exp_beta"] = float((np.exp(BETA_TRUE) * nimp).sum() / nimp.sum())
R["model_impression_weighted_mean_exp_beta"] = float((np.exp(beta_hat) * nimp).sum() / nimp.sum())
R["impressions_per_title_min_median_max"] = [
    float(nimp.min()),
    float(np.median(nimp)),
    float(nimp.max()),
]

# share of viewing that came from a recommended title (the Netflix post's own 'natural first attempt' metric)
ii, tt = np.nonzero(CHOICE < J)
in_rec = (RECS[ii, tt] == CHOICE[ii, tt][:, None]).any(1)
R["n_watch_events"] = int(len(ii))
R["share_of_viewing_from_recommended_titles"] = float(in_rec.mean())
recw = np.bincount(CHOICE[ii, tt][in_rec], minlength=J)
R["recommended_watch_events_per_title_min_median_max"] = [
    int(recw.min()),
    float(np.median(recw)),
    int(recw.max()),
]
R["n_titles_with_zero_recommended_watches"] = int((recw == 0).sum())
R["n_titles_with_lt5_recommended_watches"] = int((recw < 5).sum())

# ---------- 3. is the 'hidden' true taste a deterministic function of the history the model is given?
M = g["M"]
s_ = g["s_"]
vt_ = g["vt_"]
members = g["members"]
U_raw = (M @ vt_.T) / np.sqrt(s_)  # = u_ * sqrt(s_) by the SVD identity
U_c = U_raw - U_raw.mean(0)
Un2 = U_c / (np.linalg.norm(U_c, axis=1, keepdims=True) + 1e-9)
TASTE2 = Un2[members] @ g["Vn"].T
R["max_abs_diff_TASTE_rebuilt_from_HIST_only"] = float(np.abs(TASTE2 - g["TASTE"]).max())
R["HIST_equals_M_rows"] = bool((np.asarray(M[members].todense()) > 0).__eq__(HIST).all())

# ---------- 4. how big are the pieces of utility?
a3 = g["A_TRUE"] * g["TASTE"]
R["sd_3xaffinity_all_cells"] = float(a3.std())
R["sd_3xaffinity_within_title_across_members"] = float(a3.std(0).mean())
R["sd_0.6xpop_across_titles"] = float((g["C1"] * pop).std())
R["sd_rec_noise"] = 0.5
R["sd_V_TRUE_within_member_across_titles"] = float(V_TRUE.std(1).mean())
R["history_len_min_median_max"] = [
    int(HIST.sum(1).min()),
    float(np.median(HIST.sum(1))),
    int(HIST.sum(1).max()),
]

json.dump(R, open(os.path.join(HERE, "results", f"a_estimands_w{seed}.json"), "w"), indent=1)
print(json.dumps(R, indent=1), flush=True)

# ---------- 5. the oracle test (true taste given), done three ways
if do_oracle:
    O = {}
    # (i) the notebook's cell, verbatim
    g["BETA_HAT"] = beta_hat
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(cell_src(12), "<oracle>", "exec"), g)
    O["notebook_cell_output"] = buf.getvalue().strip()
    bo = g["beta_o"].detach().numpy().copy()
    O["notebook_mean"] = float(bo.mean())
    O["notebook_min_beta_j"] = float(bo.min())
    O["notebook_impression_weighted_mean"] = float((bo * nimp).sum() / nimp.sum())
    O["notebook_n_beta_below_0.3"] = int((bo < 0.3).sum())
    O["notebook_mean_over_titles_ge20_recwatches"] = float(bo[recw >= 20].mean())
    O["true_mean_over_titles_ge20_recwatches"] = float(BETA_TRUE[recw >= 20].mean())
    print(json.dumps(O, indent=1), flush=True)

    # (ii) exact MLE by full-batch L-BFGS (per-day chunks)
    V_T = torch.tensor(V_TRUE, dtype=torch.float64)
    hist_b = torch.tensor(HIST)
    wd_t = torch.tensor(WATCH_DAY)
    recs_t = torch.tensor(RECS)
    ch_t = torch.tensor(CHOICE)

    def nll(beta_vec):
        tot = 0.0
        for t in range(T):
            Cm = torch.zeros(I, J, dtype=torch.float64).scatter_(1, recs_t[:, t], 1.0)
            u = V_T + beta_vec * Cm
            u = u.masked_fill(hist_b | (wd_t < t), -1e9)
            logits = torch.cat([u, torch.zeros(I, 1, dtype=torch.float64)], 1)
            tot = tot + torch.nn.functional.cross_entropy(logits, ch_t[:, t], reduction="sum")
        return tot

    # common scalar beta
    b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b0], max_iter=60, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad()
        l = nll(b0.expand(J))
        l.backward()
        return l

    opt.step(clo)
    O["exact_MLE_common_scalar_beta"] = float(b0.item())
    # curvature -> s.e.
    bb = b0.detach().clone().requires_grad_(True)
    l = nll(bb.expand(J))
    (gr,) = torch.autograd.grad(l, bb, create_graph=True)
    (hs,) = torch.autograd.grad(gr.sum(), bb)
    O["exact_MLE_common_scalar_beta_se"] = float(1 / np.sqrt(hs.item()))
    print(json.dumps(O, indent=1), flush=True)

    # per-title beta with a very weak ridge (sd 2) around the scalar MLE, only to keep unidentified titles finite
    bj = torch.full((J,), b0.item(), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bj], max_iter=150, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
    c = b0.item()

    def clo2():
        opt.zero_grad()
        l = nll(bj) + ((bj - c) ** 2).sum() / (2 * 2.0**2)
        l.backward()
        return l

    opt.step(clo2)
    bjn = bj.detach().numpy()
    O["exact_MLE_per_title_mean"] = float(bjn.mean())
    O["exact_MLE_per_title_impression_weighted_mean"] = float((bjn * nimp).sum() / nimp.sum())
    O["exact_MLE_per_title_mean_over_ge20_recwatches"] = float(bjn[recw >= 20].mean())
    O["exact_MLE_per_title_min"] = float(bjn.min())
    O["exact_MLE_mean_exp"] = float(np.exp(bjn).mean())
    O["true_mean_beta"] = float(BETA_TRUE.mean())
    O["true_impression_weighted_mean_beta"] = R["true_impression_weighted_mean_beta"]
    R["oracle"] = O
    json.dump(R, open(os.path.join(HERE, "results", f"a_estimands_w{seed}.json"), "w"), indent=1)
    print(json.dumps(O, indent=1), flush=True)
