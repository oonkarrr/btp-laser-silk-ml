"""Predict laser-cut quality for any process setting, using the saved final models.

Examples (run from the project root):
  python src/predict.py --sod 5.5 --lp 15.3 --ss 300
  python src/predict.py --csv my_settings.csv          # columns: sod_mm, lp_w, ss_mm_s

For each setting it prints the predicted SLL, kerf width (Kw) and carbonisation width (Cw),
each with a 95 % range (mean +/- 1.96 sigma), and WARNS when the setting is outside
the tested range or in a region where the model is known to be weak.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def warnings_for(sod, lp, ss):
    w = []
    if not 4 <= sod <= 8:
        w.append("SOD outside tested 4-8 mm: EXTRAPOLATION, do not trust")
    if not 15.3 <= lp <= 17.1:
        w.append("LP outside tested 15.3-17.1 W: EXTRAPOLATION, do not trust")
    if not 200 <= ss <= 300:
        w.append("SS outside tested 200-300 mm/s: EXTRAPOLATION, do not trust")
    if sod > 7 and lp < 16.2:
        w.append("Through-cut may FAIL here (no cut at SOD 8 / 15.3 W)")
    if (4 <= sod < 4.5) or (7.5 < sod <= 8):
        w.append("Near edge of SOD range: lower reliability")
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict laser-cut quality (SLL, Kw, Cw).")
    ap.add_argument("--sod", type=float, help="stand-off distance, mm")
    ap.add_argument("--lp", type=float, help="laser power, W")
    ap.add_argument("--ss", type=float, help="cutting speed, mm/s")
    ap.add_argument("--csv", type=str, help="CSV file with columns sod_mm, lp_w, ss_mm_s")
    args = ap.parse_args()

    if not (MODELS / "gp_final.joblib").exists():
        sys.exit("No saved model found. Run:  python src/train_final.py")
    card = json.loads((MODELS / "model_card.json").read_text())
    if card["software"]["scikit-learn"] != sklearn.__version__:
        print(f"WARNING: model saved with scikit-learn {card['software']['scikit-learn']}, "
              f"you have {sklearn.__version__}. Re-run  python src/train_final.py\n")
    bundle = joblib.load(MODELS / "gp_final.joblib")

    if args.csv:
        settings = pd.read_csv(args.csv)[bundle["features"]]
    elif None not in (args.sod, args.lp, args.ss):
        settings = pd.DataFrame([[args.sod, args.lp, args.ss]], columns=bundle["features"])
    else:
        ap.error("give either --sod --lp --ss, or --csv")

    X = settings.to_numpy(dtype=float)
    out = settings.copy()
    for t in bundle["targets"]:
        mu, sd = bundle["models"][t].predict(X, return_std=True)
        out[f"{t}_pred"], out[f"{t}_95lo"], out[f"{t}_95hi"] = mu, mu - 1.96 * sd, mu + 1.96 * sd

    names = {"mean_sll_mm": "SLL", "mean_kw_mm": "Kerf width Kw", "mean_cw_mm": "Carbonisation width Cw"}
    for i, row in out.iterrows():
        sod, lp, ss = X[i]
        print(f"Setting: SOD = {sod:g} mm, LP = {lp:g} W, SS = {ss:g} mm/s")
        for t in bundle["targets"]:
            m, lo, hi = row[f"{t}_pred"], row[f"{t}_95lo"], row[f"{t}_95hi"]
            print(f"  {names[t]:<23} {m:.3f} mm   (95 % range {lo:.3f} - {hi:.3f} mm)")
        print(f"  {'SLL overcut':<23} {row['mean_sll_mm_pred'] - 10:.3f} mm")
        for w in warnings_for(sod, lp, ss):
            print(f"  ! {w}")
        print()

    if args.csv:
        dest = Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")
        out.round(5).to_csv(dest, index=False)
        print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
