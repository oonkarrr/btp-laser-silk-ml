"""E01 + E02: baseline and classical regression models, evaluated by LOOCV.

E01  Mean predictor    - always predicts the training-set average (the "know-nothing" reference)
E02a Linear regression - y = b0 + b1*SOD + b2*LP + b3*SS
E02b Quadratic (RSM)   - adds squares and pairwise products, e.g. SOD^2 and SOD*LP

Validation: leave-one-out cross-validation (LOOCV). With only 42 rows, a single
train/test split would leave ~8 test rows, and the score would depend heavily
on which rows were picked. LOOCV trains 42 times; each time it holds out one run,
predicts it, and then puts it back. Every run is predicted exactly once by a model
that never saw it.

Run from the project root:   python src/exp01_02_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, N_REPLICATES, ROOT, SD_COLS, TARGETS, load_cut_runs  # noqa: E402

OUT = ROOT / "reports" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# The pipelines: preprocessing + model are bundled, so scaling is re-fitted
# on the training rows only in every LOOCV round (no information leaks from the test row).
MODELS = {
    "E01_mean_baseline": DummyRegressor(strategy="mean"),
    "E02a_linear": make_pipeline(StandardScaler(), LinearRegression()),
    "E02b_quadratic_RSM": make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
    ),
}


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    loo = LeaveOneOut()
    rows, preds = [], df[["sl_no"] + FEATURES + TARGETS].copy()

    for target in TARGETS:
        y = df[target].to_numpy()
        # Noise floor (upper bound): uncertainty of a 3-replicate mean ~ SD / sqrt(3)
        noise_floor = float(df[SD_COLS[target]].mean() / np.sqrt(N_REPLICATES))
        for name, model in MODELS.items():
            y_hat = cross_val_predict(model, X, y, cv=loo)
            preds[f"pred_{name}_{target}"] = y_hat
            rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
            rows.append({
                "target": target,
                "model": name,
                "LOOCV_MAE_mm": round(float(mean_absolute_error(y, y_hat)), 5),
                "LOOCV_RMSE_mm": round(rmse, 5),
                "LOOCV_R2": round(float(r2_score(y, y_hat)), 4),
                "noise_floor_mm": round(noise_floor, 5),
                "RMSE_over_noise": round(rmse / noise_floor, 2),
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "E01_E02_loocv_results.csv", index=False)
    preds.to_csv(OUT / "E01_E02_loocv_predictions.csv", index=False)

    # Fit the quadratic model on all 42 runs only to inspect its coefficients
    # (the LOOCV numbers above are the honest performance estimate)
    coefs = {}
    for target in TARGETS:
        m = MODELS["E02b_quadratic_RSM"].fit(X, df[target])
        names = m.named_steps["polynomialfeatures"].get_feature_names_out(FEATURES)
        coefs[target] = dict(zip(names, np.round(m.named_steps["linearregression"].coef_, 5).tolist()))
    (OUT / "E02b_quadratic_coefficients_standardised.json").write_text(json.dumps(coefs, indent=2))

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
