"""
Apply the same weak-probe battery to SPX IV surfaces: does any nonlinear DR beat
PCA at predicting a forward volatility-regime move, and is any edge a dimension
artefact (as it was for LOB)?

Target: forward 5-day change in ATM-30d IV (a regime-shift signal).
Eval: TimeSeriesSplit (no look-ahead) kNN R^2 on the embedding, dims {2,3,5}.
Baselines: current ATM-30 level (persistence) and raw 63-D surface.

Run:  python experiments/iv_surface/iv_probe.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

OUT = os.path.dirname(os.path.abspath(__file__))
IV_DIR = os.path.join(OUT, "..", "..", "data", "iv")
surf = pd.read_parquet(os.path.join(IV_DIR, "surfaces.parquet")).sort_index()

atm30 = surf["T30_m1000"]
fwd = atm30.shift(-5) - atm30          # forward 5-day ATM IV change
y = fwd.to_numpy()
X = surf.values
mask = np.isfinite(y)
X, y, cur = X[mask], y[mask], atm30.to_numpy()[mask]

tscv = TimeSeriesSplit(n_splits=5)


def knn_r2(emb, k=20):
    """Walk-forward kNN: fit on each train fold, predict its test fold, pool R^2."""
    Xs = StandardScaler().fit_transform(emb)
    yt, yp = [], []
    for tr, te in tscv.split(Xs):
        reg = KNeighborsRegressor(n_neighbors=min(k, len(tr))).fit(Xs[tr], y[tr])
        yp.append(reg.predict(Xs[te]))
        yt.append(y[te])
    return float(r2_score(np.concatenate(yt), np.concatenate(yp)))


Xs_full = StandardScaler().fit_transform(X)
rows = [dict(dim=1, method="cur_atm_persistence", knn_r2=knn_r2(cur.reshape(-1, 1))),
        dict(dim=63, method="raw_surface", knn_r2=knn_r2(Xs_full))]

for dim in (2, 3, 5):
    rows.append(dict(dim=dim, method="PCA",
                     knn_r2=knn_r2(PCA(dim, random_state=0).fit_transform(Xs_full))))
    rows.append(dict(dim=dim, method="Isomap",
                     knn_r2=knn_r2(Isomap(n_neighbors=15, n_components=dim).fit_transform(Xs_full))))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "iv_probe_metrics.csv"), index=False)
print("Target: forward 5-day change in ATM-30d IV   (n =", len(y), "days)")
print(res.to_string(index=False))
print("\nNote: low/negative R^2 is expected -- forward IV *changes* are near-unpredictable;")
print("the point is the RELATIVE PCA-vs-Isomap comparison and the dimension trend.")
