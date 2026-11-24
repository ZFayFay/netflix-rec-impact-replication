"""Sample-size check: same notebook code, 16,000 members instead of 4,000 (fit cached by b_fits.py bigI)."""

import sys, os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import load

n = 16000
g = load(42, upto="sim", patches={"data": [("4000, replace=False", f"{n}, replace=False")]})
z = np.load(os.path.join(HERE, "cache", f"bigI{n}_w42_s0.npz"))
bt = g["BETA_TRUE"]
bh = z["beta"]
R = {
    "I": g["I"],
    "true_mean_beta": float(bt.mean()),
    "model_mean_beta": float(bh.mean()),
    "beta_error_pct": float(100 * (bh.mean() / bt.mean() - 1)),
    "true_mean_exp_beta": float(np.exp(bt).mean()),
    "model_mean_exp_beta": float(np.exp(bh).mean()),
    "corr_beta": float(np.corrcoef(bt, bh)[0, 1]),
    "daily_engagement": float((g["CHOICE"] < g["J"]).mean()),
    "epochs": 30,
    "note": "same code, same 30 epochs, 4x members",
}
json.dump(R, open(os.path.join(HERE, "results", f"bigI_{n}.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
