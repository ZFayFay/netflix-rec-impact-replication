"""Check B: everything that needs refits. Each task caches its fits and writes results/<task>.json
Usage: python3 b_fits.py <task> [args]
"""

import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load, fit_cached, cell_src, naive_ratio, day_sums
import torch

torch.set_num_threads(1)

task = sys.argv[1]
HOOK_PATCH = [
    (
        "            opt.zero_grad(); loss.backward(); opt.step()\n    return model",
        "            opt.zero_grad(); loss.backward(); opt.step()\n        _hook(ep, model)\n    return model",
    )
]


def save(name, R):
    json.dump(R, open(os.path.join(HERE, "results", f"{name}.json"), "w"), indent=1)
    print(json.dumps(R, indent=1), flush=True)


def world(seed=42, upto="cf", extra_patches=None, **kw):
    patches = {"fit": HOOK_PATCH}
    patches.update(extra_patches or {})
    g = load(
        seed,
        upto=upto,
        skip=("fig1", "fig2", "fig3", "fig4", "oracle"),
        defs_only=("fit", "div", "cf", "dec"),
        patches=patches,
        **kw,
    )
    g["_hook"] = lambda ep, model: None
    return g


def cf_changes(g, v, beta):
    return {k: float(x) for k, x in g["evaluate"](v, beta)["change vs current"].items()}


def stress_world(g, sigma):
    XI = np.random.default_rng(77).normal(0, 0.8, (g["I"], g["J"])) * (sigma / 0.8)
    V2 = g["V_TRUE"] + XI
    RECS2, CHOICE2, WD2, WEND2 = g["simulate"](V2, V2 + g["REC_NOISE"], np.random.default_rng(78))
    return V2, RECS2, CHOICE2, WD2


if task == "stress":
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    sigmas = (
        [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.8, 0.4, 0.2, 0.1]
    )
    g = world(seed, upto="fit")
    name = f"stress_w{seed}"
    path = os.path.join(HERE, "results", f"{name}.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    R["true_mean_exp_beta"] = float(np.exp(g["BETA_TRUE"]).mean())
    R["true_mean_beta"] = float(g["BETA_TRUE"].mean())
    R["sd_3xaffinity"] = float((g["A_TRUE"] * g["TASTE"]).std())
    for s in sigmas:
        V2, RECS2, CHOICE2, WD2 = stress_world(g, s)
        nr, r1, r0 = naive_ratio(g, RECS2, CHOICE2, WD2)
        b2, v2 = fit_cached(g, f"stress_w{seed}_sig{s}", RECS2, CHOICE2, WD2, seed=5)
        R[f"sigma_{s}"] = {
            "naive_ratio_in_that_world": nr,
            "daily_engagement": float((CHOICE2 < g["J"]).mean()),
            "truth": day_sums(g, V2, g["BETA_TRUE"], RECS2, WD2),
            "model_mean_beta": float(b2.mean()),
            "model_mean_exp_beta": float(np.exp(b2).mean()),
            "model": day_sums(g, v2, b2, RECS2, WD2),
            "hidden_share_of_member_title_taste_variance": float(
                s**2 / (s**2 + (g["A_TRUE"] * g["TASTE"]).var())
            ),
        }
        save(name, R)

elif task == "optnoise":
    g = world(42)
    R = {"truth_cf": cf_changes(g, g["V_TRUE"], g["BETA_TRUE"]), "fits": {}}
    for s in [0, 1, 2, 3, 4]:
        b, v = fit_cached(g, f"base_w42_s{s}", seed=s)
        R["fits"][s] = {
            "mean_beta": float(b.mean()),
            "mean_exp_beta": float(np.exp(b).mean()),
            "cf": cf_changes(g, v, b),
        }
        save("optnoise_w42", R)

elif task == "boot":
    seed = int(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    g = world(seed)
    name = f"boot_w{seed}_{lo}_{hi}"
    R = {"truth_cf": cf_changes(g, g["V_TRUE"], g["BETA_TRUE"]), "draws": {}}
    # truth Monte-Carlo error: re-evaluate truth with different recommendation draws for the 'current' policy
    for b in range(lo, hi):
        idx = np.random.default_rng(900 + b).choice(g["I"], g["I"], replace=True)
        bb, vb = fit_cached(g, f"boot_w{seed}_b{b}", members_idx=idx, seed=b)
        R["draws"][b] = {"mean_beta": float(bb.mean()), "cf": cf_changes(g, vb, bb)}
        save(name, R)

elif task == "encoder":
    g = world(42, upto="fit")
    R = {"true_mean_beta": float(g["BETA_TRUE"].mean())}
    nn = torch.nn
    Base = g["ChoiceModel"]
    # (a) longer training, checkpoints (same seeds/permutations as the notebook's 30-epoch fit)
    ck = {}

    def hook(ep, model):
        if (ep + 1) in (10, 20, 30, 45, 60, 90):
            ck[ep + 1] = float(model.beta.detach().mean())
            print("ckpt", ep + 1, ck[ep + 1], flush=True)

    g["_hook"] = hook
    g["fit_choice_model"](g["RECS"], g["CHOICE"], g["WATCH_DAY"], epochs=90)
    R["normalized_encoder_mean_beta_by_epoch"] = dict(ck)
    save("encoder_w42", R)

    # (b) 'plain mean-pooling encoder (no normalization)': member vector = W(mean-pooled embeddings), no norm, no scale
    class Plain(Base):
        def member_vec(self, hist):
            h = hist / hist.sum(1, keepdim=True).clamp(min=1)
            return self.W(h @ self.E)

    ck.clear()
    g["ChoiceModel"] = Plain
    g["fit_choice_model"](g["RECS"], g["CHOICE"], g["WATCH_DAY"], epochs=60)
    R["plain_meanpool_mean_beta_by_epoch"] = dict(ck)
    save("encoder_w42", R)

    # (c) sum-pooling (no division by history length) + normalization: the exact functional form of the DGP
    class SumNorm(Base):
        def member_vec(self, hist):
            a = self.W(hist @ self.E)
            return self.scale * a / a.norm(dim=1, keepdim=True).clamp(min=1e-6)

    ck.clear()
    g["ChoiceModel"] = SumNorm
    g["fit_choice_model"](g["RECS"], g["CHOICE"], g["WATCH_DAY"], epochs=30)
    R["sumpool_normalized_mean_beta_by_epoch"] = dict(ck)
    save("encoder_w42", R)

elif task == "worlds":
    for seed in [int(x) for x in sys.argv[2].split(",")]:
        g = world(seed)
        R = {
            "seed": seed,
            "naive_ratio": float(g["naive_ratio"]),
            "true_mean_exp_beta": float(g["true_ratio"]),
            "true_mean_beta": float(g["BETA_TRUE"].mean()),
            "daily_engagement": float((g["CHOICE"] < g["J"]).mean()),
        }
        b, v = fit_cached(g, f"base_w{seed}_s0")
        R["model_mean_beta"] = float(b.mean())
        R["model_mean_exp_beta"] = float(np.exp(b).mean())
        R["beta_error_pct"] = 100 * (R["model_mean_beta"] / R["true_mean_beta"] - 1)
        R["truth_prob"] = day_sums(g, g["V_TRUE"], g["BETA_TRUE"], g["RECS"], g["WATCH_DAY"])
        R["model_prob"] = day_sums(g, v, b, g["RECS"], g["WATCH_DAY"])
        R["truth_cf"] = cf_changes(g, g["V_TRUE"], g["BETA_TRUE"])
        R["model_cf"] = cf_changes(g, v, b)
        save(f"world_{seed}", R)
        # notebook's oracle cell verbatim
        g["BETA_HAT"] = b
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(cell_src(12), "<oracle>", "exec"), g)
        R["notebook_oracle_output"] = buf.getvalue().strip()
        save(f"world_{seed}", R)
        # stress at sigma 0.8 (same construction as notebook cell 23)
        V2, RECS2, CHOICE2, WD2 = stress_world(g, 0.8)
        nr, _, _ = naive_ratio(g, RECS2, CHOICE2, WD2)
        b2, v2 = fit_cached(g, f"stress_w{seed}_sig0.8", RECS2, CHOICE2, WD2, seed=5)
        R["stress_0.8"] = {
            "naive_ratio_in_that_world": nr,
            "model_mean_beta": float(b2.mean()),
            "model_mean_exp_beta": float(np.exp(b2).mean()),
            "truth": day_sums(g, V2, g["BETA_TRUE"], RECS2, WD2),
        }
        save(f"world_{seed}", R)

elif task == "bigI":
    n = int(sys.argv[2])
    g = world(
        42, upto="fit", extra_patches={"data": [("4000, replace=False", f"{n}, replace=False")]}
    )
    R = {
        "I": g["I"],
        "naive_ratio": float(g["naive_ratio"]),
        "true_mean_beta": float(g["BETA_TRUE"].mean()),
        "true_mean_exp_beta": float(g["true_ratio"]),
        "daily_engagement": float((g["CHOICE"] < g["J"]).mean()),
    }
    ck = {}

    def hook(ep, model):
        if (ep + 1) in (10, 20, 30):
            ck[ep + 1] = float(model.beta.detach().mean())
            print("ckpt", ep + 1, ck[ep + 1], flush=True)

    g["_hook"] = hook
    t0 = time.time()
    b, v = fit_cached(g, f"bigI{n}_w42_s0")
    R["model_mean_beta"] = float(b.mean())
    R["model_mean_exp_beta"] = float(np.exp(b).mean())
    R["beta_error_pct"] = 100 * (R["model_mean_beta"] / R["true_mean_beta"] - 1)
    R["by_epoch"] = ck
    R["fit_seconds"] = time.time() - t0
    save(f"bigI_{n}", R)
