"""Train the FINAL models on all 42 cut runs and save them to models/.

What is saved (models/gp_final.joblib): for each target (SLL, Kw, Cw) a pipeline
  StandardScaler -> GaussianProcessRegressor (RBF + white noise; chosen in E08)
plus models/model_card.json describing exactly what the models are, what data
they were trained on, how well they performed in validation, and where they are
valid. Anyone (your guide, an examiner, future you) can check the model from this file.

Why train on ALL the data now? Cross-validation (E03-E08) was for ESTIMATING how well
this recipe works on unseen settings. Once the recipe is fixed, the best final model
uses every run. The validation numbers in the model card describe this recipe.

Run from the project root:   python src/train_final.py
(Re-run it whenever you change the data or the library versions.)
"""
import hashlib
import json
import platform
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, RAW_CSV, ROOT, TARGETS, load_cut_runs  # noqa: E402
from exp05_gp_uncertainty import make_gp  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
MODELS = ROOT / "models"

# Validation results of this exact recipe (copied from verified experiments; see the log)
VALIDATION = {
    "mean_sll_mm": {"cellout_R2": 0.9018, "cellout_RMSE_mm": 0.02305, "coverage_95": 0.905},
    "mean_kw_mm": {"cellout_R2": 0.9299, "cellout_RMSE_mm": 0.01752, "coverage_95": 0.929},
    "mean_cw_mm": {"cellout_R2": 0.9490, "cellout_RMSE_mm": 0.00438, "coverage_95": 0.976},
}


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    bundle, card_targets = {}, {}
    for t in TARGETS:
        pipe = make_pipeline(StandardScaler(), make_gp()).fit(X, df[t])
        bundle[t] = pipe
        gp = pipe[-1]
        y_fit = pipe.predict(X)
        card_targets[t] = {
            "learned_kernel": str(gp.kernel_),
            "training_RMSE_mm": round(float(np.sqrt(np.mean((df[t] - y_fit) ** 2))), 5),
            "validation": VALIDATION[t],
        }
        print(f"{t}: trained. Training RMSE = {card_targets[t]['training_RMSE_mm']} mm "
              f"(validation RMSE on unseen cells = {VALIDATION[t]['cellout_RMSE_mm']} mm)")

    joblib.dump({"features": FEATURES, "targets": TARGETS, "models": bundle}, MODELS / "gp_final.joblib")
    card = {
        "name": "Laser-cut quality GP models (Bombyx mori / pat silk, CO2 laser)",
        "trained_on": date.today().isoformat(),
        "data_file": str(RAW_CSV.relative_to(ROOT)),
        "data_sha256": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
        "n_training_runs": int(len(df)),
        "inputs": {"sod_mm": [4, 8], "lp_w": [15.3, 17.1], "ss_mm_s": [200, 300]},
        "valid_only_inside_input_ranges": True,
        "known_weak_regions": [
            "SOD near 4 or 8 mm (edge of data; E04 errors up to about 3x larger)",
            "LP between tested levels 15.3/16.2/17.1 W (no data; wider sigma)",
            "SOD > 7 mm with LP < 16.2 W (through-cut may fail; run 37-39 failed)",
        ],
        "outputs": {"mean_sll_mm": "Side line length (nominal 10 mm)",
                    "mean_kw_mm": "Kerf width", "mean_cw_mm": "Carbonisation width"},
        "targets": card_targets,
        "validation_method": "Leave-one-cell-out CV (all 3 speeds of one SOD-LP combination held out); see log D08",
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
    }
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nSaved: {MODELS / 'gp_final.joblib'}\nSaved: {MODELS / 'model_card.json'}")


if __name__ == "__main__":
    main()
