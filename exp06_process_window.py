"""E06: process-window map and robust optimisation.

Goal (productivity): run at the highest tested cutting speed (300 mm/s) and choose
SOD and LP so that cut quality stays good EVEN IF things drift a little.

How:
1. Fit the final Gaussian process for SLL, Kw, Cw on all 42 cut runs (E03b settings).
2. Predict on a fine grid of SOD (4-8 mm) x LP (15.3-17.1 W), at SS = 300 mm/s.
3. "Pessimistic" value at each grid point = predicted mean + K_SIGMA * sigma
   (E05 showed sigma is roughly honest, so mean + 2 sigma is a ~97.5 % one-sided bound).
4. Robustness to SOD drift (fabric wrinkling): take the WORST pessimistic value over
   SOD in [s - DRIFT, s + DRIFT]. A setting is only as good as its worst nearby case.
5. Through-cut risk: SOD 8 / LP 15.3 failed to cut. The failure boundary lies somewhere
   between that cell and its successful neighbours (SOD 7 / LP 15.3 and SOD 8 / LP 16.2),
   so the rectangle SOD > 7 AND LP < 16.2 is marked "at risk" and excluded.
   Settings whose drift window touches it, or leaves the tested SOD range, are also excluded.

ASSUMPTIONS (to confirm with the guide - log Q7/Q8): DRIFT = 0.5 mm, K_SIGMA = 2.
No quality limits are imposed here; the maps show the robust worst-case values so the
guide can read off the feasible window for any limit.

Run from the project root:   python src/exp06_process_window.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, ROOT, TARGETS, load_cut_runs, load_raw  # noqa: E402
from exp05_gp_uncertainty import make_gp  # noqa: E402  (same GP as E03b/E05)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT, FIG = ROOT / "reports" / "results", ROOT / "reports" / "figures"
SS_RUN, DRIFT, K_SIGMA = 300.0, 0.5, 2.0
SOD_GRID = np.round(np.arange(4.0, 8.0001, 0.05), 2)
LP_GRID = np.round(np.arange(15.3, 17.1001, 0.01), 2)
LABEL = {"mean_sll_mm": "SLL overcut (SLL - 10 mm)", "mean_kw_mm": "Kerf width Kw",
         "mean_cw_mm": "Carbonisation width Cw"}


def at_risk(sod, lp):
    return (sod > 7.0) & (lp < 16.2)


def main() -> None:
    df = load_cut_runs()
    X = df[FEATURES].to_numpy()
    scaler = StandardScaler().fit(X)
    S, P = np.meshgrid(SOD_GRID, LP_GRID, indexing="ij")          # shape (n_sod, n_lp)
    grid = np.column_stack([S.ravel(), P.ravel(), np.full(S.size, SS_RUN)])

    pess, mean = {}, {}
    for t in TARGETS:
        gp = make_gp().fit(scaler.transform(X), df[t])
        mu, sd = gp.predict(scaler.transform(grid), return_std=True)
        if t == "mean_sll_mm":
            mu = mu - 10.0                                          # work with overcut
        mean[t] = mu.reshape(S.shape)
        pess[t] = (mu + K_SIGMA * sd).reshape(S.shape)
        # Check: does SS change the prediction? (compare 200 vs 300 mm/s on the grid)
        g200 = grid.copy(); g200[:, 2] = 200.0
        diff = np.abs(gp.predict(scaler.transform(g200)) - gp.predict(scaler.transform(grid)))
        print(f"{t}: max |prediction(SS=200) - prediction(SS=300)| over grid = {diff.max():.5f} mm")

    # Robust value = worst pessimistic value within +/- DRIFT in SOD
    step = int(round(DRIFT / 0.05))
    robust, valid = {}, np.zeros(S.shape, bool)
    risk = at_risk(S, P)
    for i in range(len(SOD_GRID)):
        lo, hi = i - step, i + step
        if lo < 0 or hi >= len(SOD_GRID):
            continue                                                # drift would leave tested range
        valid[i] = ~risk[lo:hi + 1].any(axis=0)
    for t in TARGETS:
        r = np.full(S.shape, np.nan)
        for i in range(len(SOD_GRID)):
            lo, hi = i - step, i + step
            if 0 <= lo and hi < len(SOD_GRID):
                r[i] = pess[t][lo:hi + 1].max(axis=0)
        robust[t] = np.where(valid, r, np.nan)

    # Robust optimum per response, and a combined optimum (minimise the sum of
    # each robust value divided by its best achievable robust value)
    rows = []
    for t in TARGETS:
        i, j = np.unravel_index(np.nanargmin(robust[t]), S.shape)
        rows.append({"objective": f"minimise robust {t}", "SOD_mm": SOD_GRID[i], "LP_W": LP_GRID[j]})
    score = sum(robust[t] / np.nanmin(robust[t]) for t in TARGETS)
    i, j = np.unravel_index(np.nanargmin(score), S.shape)
    rows.append({"objective": "combined (all three)", "SOD_mm": SOD_GRID[i], "LP_W": LP_GRID[j]})
    for r in rows:
        ii, jj = list(SOD_GRID).index(r["SOD_mm"]), list(LP_GRID).index(r["LP_W"])
        for t in TARGETS:
            r[f"{t}_pred_mean"] = round(float(mean[t][ii, jj]), 4)
            r[f"{t}_robust_bound"] = round(float(robust[t][ii, jj]), 4)
    opt = pd.DataFrame(rows)
    opt.insert(3, "SS_mm_s", SS_RUN)
    opt.to_csv(OUT / "E06_robust_optimum.csv", index=False)
    print("\nRobust optima at SS = 300 mm/s (DRIFT = 0.5 mm, mean + 2 sigma):")
    print(opt.to_string(index=False))

    # Nearest measured runs for comparison (the honest check without new experiments)
    raw = load_raw()
    near = raw[(raw.lp_w == 15.3) & (raw.sod_mm.isin([5, 6])) & (raw.ss_mm_s == 300)]
    print("\nNearest measured runs at 300 mm/s:")
    print(near[["sl_no", "sod_mm", "lp_w", "ss_mm_s", "mean_sll_mm", "mean_kw_mm", "mean_cw_mm"]]
          .to_string(index=False))

    # Window size vs limit (for the guide to pick limits): share of the valid SOD-LP area
    # where the robust bound is below a given limit
    sweep = []
    for t in TARGETS:
        v = robust[t][~np.isnan(robust[t])]
        for lim in np.round(np.linspace(np.nanmin(robust[t]), np.nanmax(robust[t]), 12), 3):
            sweep.append({"target": t, "limit_mm": lim, "share_of_window_feasible": round(float((v <= lim).mean()), 3)})
    pd.DataFrame(sweep).to_csv(OUT / "E06_window_vs_limit.csv", index=False)

    # Figure: robust worst-case maps
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.15))
    for a, t in zip(ax, TARGETS):
        cs = a.contourf(P, S, robust[t], levels=14, cmap="viridis")
        a.contour(P, S, robust[t], levels=7, colors="white", linewidths=0.5)
        fig.colorbar(cs, ax=a, label="robust worst case (mm)")
        a.contourf(P, S, risk.astype(float), levels=[0.5, 1.5], hatches=["//"], colors="none")
        a.scatter(raw.lp_w, raw.sod_mm, c=np.where(raw.through_cut == 1, "white", "red"),
                  edgecolors="k", s=16, zorder=3)
        a.plot(opt.LP_W.iloc[-1], opt.SOD_mm.iloc[-1], marker="*", ms=13, c="orange",
               mec="k", zorder=4)
        a.set_xlabel("Laser power LP (W)"); a.set_ylabel("Stand-off distance SOD (mm)")
        a.set_title(LABEL[t], fontsize=9.5)
    fig.suptitle("E06: robust worst case (mean + 2σ, SOD drift ±0.5 mm) at SS = 300 mm/s.\n"
                 "Blank = excluded; hatched = through-cut risk; red = no cut; ★ = combined optimum",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIG / "fig05_E06_process_window.png", dpi=220)


if __name__ == "__main__":
    main()
