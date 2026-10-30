"""Load the laser-cutting dataset (Table 5.4, Bombyx mori / pat silk).

Every experiment script imports from here, so the data is always read
and filtered the same way.
"""
from pathlib import Path

import pandas as pd

# Project root = the folder that contains src/ (works on any computer)
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "pat_silk_table_5_4.csv"

# Input features (process parameters) and output targets (quality responses)
FEATURES = ["sod_mm", "lp_w", "ss_mm_s"]
TARGETS = ["mean_sll_mm", "mean_kw_mm", "mean_cw_mm"]
# Matching SD columns (spread of the repeat measurements behind each mean)
SD_COLS = {"mean_sll_mm": "sd_sll_mm", "mean_kw_mm": "sd_kw_mm", "mean_cw_mm": "sd_cw_mm"}
N_REPLICATES = 3  # each parameter combination was cut 3 times (student, S01)


def load_raw() -> pd.DataFrame:
    """All 45 runs, including the 3 runs with no through-cut."""
    return pd.read_csv(RAW_CSV)


def load_cut_runs() -> pd.DataFrame:
    """Only the 42 runs that cut through and have measured responses."""
    df = load_raw()
    return df[df["through_cut"] == 1].reset_index(drop=True)
