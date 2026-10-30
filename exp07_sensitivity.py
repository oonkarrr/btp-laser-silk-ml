"""E07: how much does the recommended setting depend on our assumptions?

E06 assumed SOD drift = +/-0.5 mm and a safety margin of K = 2 sigma.
Both are assumptions (log Q8). Here we repeat the E06 robust optimisation for
  DRIFT in {0.0, 0.25, 0.5, 0.75, 1.0} mm   and   K in {0, 1, 2, 3}
and record where the combined optimum lands and how good its worst case is.
If the optimum barely moves, the recommendation is robust to our assumptions.

Run from the project root:   python src/exp07_sensitivity.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, ROOT, TARGETS, load_cut_runs  # noqa: E402
from exp05_gp_uncertainty import make_gp  # noqa: E402
from exp06_process_window import LP_GRID, SOD_GRID, SS_RUN, at_risk  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = ROOT / "reports" / "results"
STEP_MM = 0.05  # SOD grid spacing


def robust_optimum(mean, sd, drift, k, risk):
    """Combined robust optimum for one (drift, k) pair. Returns a dict, or None if no valid point."""
    step = int(round(drift / STEP_MM))
    n = len(SOD_GRID)
    robust = {}
    valid = np.zeros(risk.shape, bool)
    for i in range(step, n - step):
        valid[i] = ~risk[i - step:i + step + 1].any(axis=0)
    for t in TARGETS:
        pess = mean[t] + k * sd[t]
        r = np.full(risk.shape, np.nan)
        for i in range(step, n - step):
            r[i] = pess[i - step:i + step + 1].max(axis=0)
        robust[t] = np.where(valid, r, np.nan)
    if np.all(np.isnan(robust[TARGETS[0]])):
        return None
    score = sum(robust[t] / np.nanmin(robust[t]) for t in TARGETS)
    i, j = np.unravel_index(np.nanargmin(score), risk.shape)
    out = {"DRIFT_mm": drift, "K_sigma": k, "SOD_mm": SOD_GRID[i], "LP_W": LP_GRID[j]}
    for t in TARGETS:
        out[f"{t}_robust_bound"] = round(float(robust[t][i, j]), 4)
    return out


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    scaler = StandardScaler().fit(X)
    S, P = np.meshgrid(SOD_GRID, LP_GRID, indexing="ij")
    grid = np.column_stack([S.ravel(), P.ravel(), np.full(S.size, SS_RUN)])
    mean, sd = {}, {}
    for t in TARGETS:
        gp = make_gp().fit(scaler.transform(X), df[t])
        mu, s = gp.predict(scaler.transform(grid), return_std=True)
        mean[t] = (mu - 10.0 if t == "mean_sll_mm" else mu).reshape(S.shape)
        sd[t] = s.reshape(S.shape)
    risk = at_risk(S, P)

    rows = []
    for drift in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for k in [0, 1, 2, 3]:
            r = robust_optimum(mean, sd, drift, k, risk)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E07_sensitivity.csv", index=False)
    print(res.to_string(index=False))
    print(f"\nOptimum SOD range over all assumptions: {res.SOD_mm.min()} - {res.SOD_mm.max()} mm")
    print(f"Optimum LP  range over all assumptions: {res.LP_W.min()} - {res.LP_W.max()} W")


if __name__ == "__main__":
    main()
