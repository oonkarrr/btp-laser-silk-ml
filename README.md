# Data

The raw experimental data (`data/raw/pat_silk_table_5_4.csv`) were produced by the supervisor's research group and are **not included** in this repository.

To re-run the experiments in `src/` you need that file with these columns: `sl_no, sod_mm, lp_w, ss_mm_s, through_cut, mean_sll_mm, sd_sll_mm, mean_kw_mm, sd_kw_mm, mean_cw_mm, sd_cw_mm` (45 rows, one per parameter combination; each value is the mean of three cuts).

The trained model in `models/` and `src/predict.py` do **not** need the data.
