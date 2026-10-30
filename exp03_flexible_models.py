"""E03: can more flexible models capture what the quadratic RSM missed?

E02 showed the quadratic model's errors are systematic: the shape of the SOD curve
changes with laser power (flat at 15.3 W, a deep V at 17.1 W). E03 tests four
candidates, using exactly the same LOOCV procedure as E02 so the scores are comparable.

E03a Extended RSM     - quadratic + SOD^2*LP and SOD*LP^2 terms (lets the SOD curve bend differently per power)
E03b Gaussian process - a smooth, flexible curve-fitter that also reports its own uncertainty
E03c Random forest    - average of many decision trees (step-shaped predictions)
E03d Gradient boosting- trees added one by one, each correcting the previous errors

Two validation schemes are reported:
  LOOCV          - hold out one run at a time (as in E02)
  LOCO (cell-out)- hold out all 3 speeds of one (SOD, LP) combination at once.
                   Because SS has ~no effect, a run's two "siblings" at other speeds are
                   almost repeat measurements of it; LOOCV lets the model copy them.
                   LOCO asks the harder, more honest question: can the model predict a
                   (SOD, LP) combination it has never seen?

Hyperparameters are FIXED in advance (not tuned on these LOOCV scores), so the
comparison stays honest. The Gaussian process learns its length-scales inside each
training fold by maximising likelihood - that uses training rows only, so there is no leakage.

Run from the project root:   python src/exp03_flexible_models.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, N_REPLICATES, ROOT, SD_COLS, TARGETS, load_cut_runs  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42  # fixes the randomness in the tree models so results are repeatable
OUT = ROOT / "reports" / "results"


class ExtraInteractions(BaseEstimator, TransformerMixin):
    """Adds SOD^2*LP and SOD*LP^2 to the inputs (columns 0 = SOD, 1 = LP after scaling)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sod, lp = X[:, [0]], X[:, [1]]
        return np.hstack([X, sod**2 * lp, sod * lp**2])


def build_models():
    # GP kernel: overall size x smooth RBF with one length-scale per input
    # (so it can learn that SS hardly matters) + a white-noise term for measurement noise
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
    return {
        "E02b_quadratic_RSM (reference)": make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
        "E03a_extended_RSM": make_pipeline(
            StandardScaler(), ExtraInteractions(), PolynomialFeatures(2, include_bias=False),
            LinearRegression()),
        "E03b_gaussian_process": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                     n_restarts_optimizer=5, random_state=SEED)),
        "E03c_random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=SEED),
        "E03d_gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, random_state=SEED),
    }


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    cells = (df["sod_mm"].astype(str) + "_" + df["lp_w"].astype(str)).to_numpy()  # 14 (SOD, LP) groups
    schemes = {"LOOCV": (LeaveOneOut(), None), "LOCO_cell_out": (LeaveOneGroupOut(), cells)}
    rows, preds = [], df[["sl_no"] + FEATURES + TARGETS].copy()
    for scheme, (cv, groups) in schemes.items():
        for target in TARGETS:
            y = df[target].to_numpy()
            noise = float(df[SD_COLS[target]].mean() / np.sqrt(N_REPLICATES))
            for name, model in build_models().items():
                y_hat = cross_val_predict(model, X, y, cv=cv, groups=groups)
                preds[f"{scheme}_{name.split()[0]}_{target}"] = y_hat
                rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
                rows.append({"scheme": scheme, "target": target, "model": name,
                             "MAE_mm": round(float(mean_absolute_error(y, y_hat)), 5),
                             "RMSE_mm": round(rmse, 5),
                             "R2": round(float(r2_score(y, y_hat)), 4),
                             "noise_floor_mm": round(noise, 5),
                             "RMSE_over_noise": round(rmse / noise, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E03_cv_results.csv", index=False)
    preds.to_csv(OUT / "E03_cv_predictions.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
