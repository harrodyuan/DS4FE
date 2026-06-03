DS4FE — SPX Implied-Volatility Surface Reconstruction
=====================================================

Contents of this package
-------------------------
  DS4FE_SPX_IV_Surface_Reconstruction_from_Raw_EOD_Quotes.ipynb
      The notebook (already executed; outputs included).

  experiments/iv_surface/iv_lib.py
      Helper module the notebook imports: Black-76 pricing and the
      implied-volatility root-finder.

  data/iv/
      iv_YYYY-MM-DD.parquet        523 files — per-option reconstructed IV table,
                                   one file per trading day (2021-06-01 .. 2023-06-30).
      surfaces.parquet             The final 523 x 63 matrix (one fixed 7x9 grid per day).
      surface_grid.json            The 7 maturities and 9 moneyness levels of the grid.
      cbbo_2023-06-01_1545.parquet Raw OPRA bid/ask snapshot for the single worked example day.
      def_2023-06-01.parquet       Contract definitions (strike/expiry/call-put) for that day.

Note: the full raw dataset has a cbbo_* and def_* file for every day (~273 MB).
Only the one example day's raw files are included here because that is all the
notebook reads directly; the 523 iv_*.parquet files already contain the
reconstructed result for every day.

How to run
----------
1. Unzip, keeping the folder structure intact.
2. From the unzipped folder, launch Jupyter and open the .ipynb.
3. Run all cells. Requirements: python 3, numpy, pandas, matplotlib, scipy, pyarrow.

What the notebook does (one line)
---------------------------------
Raw call/put mid prices -> put-call parity OLS (per expiry) -> forward F and rate r
-> invert Black-76 on each mid (1-D root-find) -> implied vol -> interpolate onto a
fixed 7x9 (maturity x moneyness) grid. No external rate/dividend/vol data is used.
