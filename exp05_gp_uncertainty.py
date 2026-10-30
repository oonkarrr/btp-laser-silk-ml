"""E05: can we trust the Gaussian process's uncertainty?

A GP gives every prediction a mean AND a standard deviation (sigma). If those sigmas
are honest, then on unseen data:
  ~68 % of true values should fall inside  mean +/- 1.00 sigma
  ~95 % of true values should fall inside  mean +/- 1.96 sigma
This is called "calibration". We test it with the same cell-out CV as E03
(hold out all 3 speeds of one SOD-LP combination, predict them, repeat for all 14).

Part 2 fits the GP on all 42 runs and prints the learned length-scales.
A length-scale is "how far you must move an input before the output changes much"
(in standardised units). A huge length-scale means that input hardly matters.

Run from the project root:   python src/exp05_gp_uncertainty.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, ROOT, TARGETS, load_cut_runs  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
OUT = ROOT / "reports" / "results"


def make_gp():
    # Same kernel as E03b
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=5, random_state=SEED)


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    cells = (df["sod_mm"].astype(str) + "_" + df["lp_w"].astype(str)).to_numpy()
    rows, per_run = [], df[["sl_no"] + FEATURES].copy()

    for target in TARGETS:
        y = df[target].to_numpy()
        mu, sd = np.zeros_like(y), np.zeros_like(y)
        for tr, te in LeaveOneGroupOut().split(X, y, cells):
            scaler = StandardScaler().fit(X[tr])          # fitted on training rows only
            gp = make_gp().fit(scaler.transform(X[tr]), y[tr])
            mu[te], sd[te] = gp.predict(scaler.transform(X[te]), return_std=True)
        z = (y - mu) / sd                                  # error measured in sigmas
        per_run[f"{target}_pred"], per_run[f"{target}_sigma"], per_run[f"{target}_z"] = mu, sd, z
        rows.append({
            "target": target,
            "coverage_68pct_interval": round(float(np.mean(np.abs(z) <= 1.0)), 3),
            "coverage_95pct_interval": round(float(np.mean(np.abs(z) <= 1.96)), 3),
            "mean_sigma_mm": round(float(sd.mean()), 5),
            "RMSE_mm": round(float(np.sqrt(np.mean((y - mu) ** 2))), 5),
            "worst_abs_z": round(float(np.abs(z).max()), 2),
            "worst_run_sl_no": int(df["sl_no"].iloc[np.abs(z).argmax()]),
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E05_gp_calibration.csv", index=False)
    per_run.to_csv(OUT / "E05_gp_cellout_predictions.csv", index=False)
    print("Calibration under cell-out CV (ideal: 0.68 and 0.95)")
    print(res.to_string(index=False))

    # Part 2: learned length-scales on all data (standardised inputs)
    scales = {}
    for target in TARGETS:
        gp = make_gp().fit(StandardScaler().fit_transform(X), df[target])
        ls = gp.kernel_.k1.k2.length_scale
        scales[target] = {f: round(float(v), 3) for f, v in zip(FEATURES, ls)}
        scales[target]["noise_sd_mm"] = round(
            float(np.sqrt(gp.kernel_.k2.noise_level) * df[target].std()), 5)
    (OUT / "E05_gp_length_scales.json").write_text(json.dumps(scales, indent=2))
    print("\nLearned length-scales (standardised units; large = input matters little)")
    print(pd.DataFrame(scales).T.to_string())


if __name__ == "__main__":
    main()
