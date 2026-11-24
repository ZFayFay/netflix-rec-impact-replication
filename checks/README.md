# Checks

Extra runs that test the notebook's results. Each script reuses the notebook's own code cells through `harness.py`, so nothing is reimplemented. Fits are cached in `cache/` and results are written to `results/`.

| Script | What it checks | Output |
|---|---|---|
| `a_estimands.py` | What the 7.5×, 3.1× and 3.4× numbers measure, on the same scale | `results/a_estimands_w42.json` |
| `b_fits.py worlds 43,44` | The whole simulation in two more random worlds | `results/world_43.json`, `results/world_44.json` |
| `b_fits.py stress 42 0.8,0.4,0.2,0.1` | The hidden-signal test at four strengths | `results/stress_w42.json` |
| `b_fits.py encoder` | Bias by history encoder and training length | `results/encoder_w42.json` |
| `b_fits.py bigI 16000` and `f_bigI_summary.py` | 16,000 members instead of 4,000 | `results/bigI_16000.json` |
| `c_diversion.py` | The nudge validation by experiment length, and a model with no personalization | `results/c_diversion.json` |
| `e_oracle.py` | The estimator given the true tastes, fit to convergence | `results/e_oracle_w42.json` |

Run from this folder, for example `python b_fits.py worlds 43,44`. The fits take from a few minutes to about an hour each on a laptop CPU.
