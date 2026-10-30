"""E08: choosing the Gaussian process kernel properly.

The kernel is the GP's built-in assumption about how smooth the response surface is.
So far we used RBF (very smooth). Alternatives:
  Matern nu=2.5 - smooth but allows sharper bends
  Matern nu=1.5 - rougher still
  RationalQuadratic (RQ) - a mixture of smoothness scales

The trap: if we try 4 kernels and keep whichever scores best on the cell-out CV,
that score becomes optimistic (we chose using the test folds). The proper way is
NESTED selection:
  outer loop: cell-out CV (hold out one SOD-LP combination), used ONLY for scoring
  inside each outer training set: choose the kernel by log-marginal-likelihood (LML),
  a goodness-of-fit measure that uses training data only and penalises over-complex fits.
The score of this whole "select-then-predict" procedure is an honest estimate.

We also report each kernel's outer score on its own (for information, not for choosing).

Run from the project root:   python src/exp08_kernel_selection.py      (~2-4 min)
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, ROOT, TARGETS, load_cut_runs  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
OUT = ROOT / "reports" / "results"
LS = dict(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e3))


def kernels():
    noise = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
    amp = lambda: ConstantKernel(1.0, (1e-3, 1e3))  # noqa: E731
    return {
        "RBF": amp() * RBF(**LS) + noise,
        "Matern_2.5": amp() * Matern(nu=2.5, **LS) + noise,
        "Matern_1.5": amp() * Matern(nu=1.5, **LS) + noise,
        # RQ in scikit-learn has a single (shared) length-scale
        "RationalQuadratic": amp() * RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                       length_scale_bounds=(1e-2, 1e3)) + noise,
    }


def fit_gp(kernel, X, y):
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                                    random_state=SEED).fit(X, y)


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    cells = (df["sod_mm"].astype(str) + "_" + df["lp_w"].astype(str)).to_numpy()
    rows, chosen_all = [], {}
    for target in TARGETS:
        y = df[target].to_numpy()
        pred = {name: np.zeros_like(y) for name in kernels()}
        pred["nested_LML_selection"] = np.zeros_like(y)
        chosen = []
        for tr, te in LeaveOneGroupOut().split(X, y, cells):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            fitted = {name: fit_gp(k, Xtr, y[tr]) for name, k in kernels().items()}
            for name, gp in fitted.items():
                pred[name][te] = gp.predict(Xte)
            best = max(fitted, key=lambda n: fitted[n].log_marginal_likelihood_value_)
            chosen.append(best)
            pred["nested_LML_selection"][te] = fitted[best].predict(Xte)
        chosen_all[target] = dict(Counter(chosen))
        for name, p in pred.items():
            rows.append({"target": target, "kernel": name,
                         "cellout_RMSE_mm": round(float(np.sqrt(np.mean((y - p) ** 2))), 5),
                         "cellout_R2": round(float(r2_score(y, p)), 4)})
        # Which kernel wins by LML when fitted on ALL data (this becomes the final choice)
        sc = StandardScaler().fit(X)
        lml = {n: fit_gp(k, sc.transform(X), y).log_marginal_likelihood_value_ for n, k in kernels().items()}
        chosen_all[target]["full_data_LML_winner"] = max(lml, key=lml.get)
        chosen_all[target]["full_data_LML"] = {n: round(v, 2) for n, v in lml.items()}

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E08_kernel_selection.csv", index=False)
    pd.Series(chosen_all).to_json(OUT / "E08_kernel_choices.json", indent=2)
    print(res.pivot(index="kernel", columns="target", values="cellout_R2").to_string())
    print("\nKernel chosen by LML inside the 14 outer training sets, and on all data:")
    for t, c in chosen_all.items():
        print(t, {k: v for k, v in c.items() if k != "full_data_LML"})


if __name__ == "__main__":
    main()
