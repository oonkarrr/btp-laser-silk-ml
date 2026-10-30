"""E04: stress test - can a model predict an SOD level it has never seen?

Hold out ALL runs at one SOD level (e.g. every run at SOD = 6 mm), train on the
other four levels, predict the held-out level. Repeat for each of the 5 levels.

SOD = 5, 6, 7  -> interpolation (there is data on both sides)
SOD = 4, 8     -> extrapolation (no data beyond) - expected to be much worse

This is harder than the cell-out test (E03) because a whole row of the design
disappears. It tells us how far we can trust the model between and beyond SOD levels.

Run from the project root:   python src/exp04_sod_level_out.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, ROOT, TARGETS, load_cut_runs  # noqa: E402
from exp03_flexible_models import build_models  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = ROOT / "reports" / "results"
KEEP = ["E02b_quadratic_RSM (reference)", "E03b_gaussian_process", "E03d_gradient_boosting"]


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    rows = []
    for target in TARGETS:
        y = df[target].to_numpy()
        for name in KEEP:
            for sod in sorted(df["sod_mm"].unique()):
                test = (df["sod_mm"] == sod).to_numpy()
                model = build_models()[name].fit(X[~test], y[~test])
                rmse = float(np.sqrt(mean_squared_error(y[test], model.predict(X[test]))))
                rows.append({"target": target, "model": name.split()[0], "held_out_SOD_mm": sod,
                             "kind": "interpolation" if 4 < sod < 8 else "extrapolation",
                             "RMSE_mm": round(rmse, 5)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E04_sod_level_out.csv", index=False)
    table = res.pivot_table(index=["target", "model"], columns="held_out_SOD_mm", values="RMSE_mm")
    print("RMSE (mm) when the whole SOD level is held out:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
