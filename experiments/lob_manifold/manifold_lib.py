"""
Manifold-learning helpers for the LOB DR study.

Provides:
  - DiffusionMap : fit on a landmark set + Nystrom out-of-sample transform,
    matching the sklearn-style .fit/.transform API used for PCA/Isomap/UMAP.
  - geometry metrics: trustworthiness wrapper, geodesic-distance preservation,
    linear reconstruction R^2.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.manifold import trustworthiness as _trustworthiness
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors


class DiffusionMap:
    """Anisotropic diffusion map (Coifman & Lafon) with Nystrom extension.

    Parameters
    ----------
    n_components : number of non-trivial diffusion coordinates to keep.
    alpha        : anisotropic normalization (1.0 ~ Laplace-Beltrami, density-free).
    epsilon      : kernel bandwidth. If None, set to the median squared pairwise
                   distance of the training landmarks (a robust default).
    t            : diffusion time (eigenvalue exponent).
    """

    def __init__(self, n_components=2, alpha=1.0, epsilon=None, t=1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.epsilon = epsilon
        self.t = t

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.X_fit_ = X
        D2 = squareform(pdist(X, "sqeuclidean"))
        if self.epsilon is None:
            med = np.median(D2[D2 > 0])
            self.epsilon_ = med if med > 0 else 1.0
        else:
            self.epsilon_ = float(self.epsilon)

        K = np.exp(-D2 / self.epsilon_)
        # anisotropic normalization
        q = K.sum(axis=1)
        self.q_fit_ = q
        Ka = K / np.outer(q ** self.alpha, q ** self.alpha)
        d = Ka.sum(axis=1)
        self.d_fit_ = d
        # symmetric conjugate of the row-stochastic operator
        d_inv_sqrt = 1.0 / np.sqrt(d)
        Ms = (d_inv_sqrt[:, None] * Ka) * d_inv_sqrt[None, :]
        Ms = 0.5 * (Ms + Ms.T)
        vals, vecs = np.linalg.eigh(Ms)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # eigenvectors of P = D^{-1/2} Ms D^{1/2}
        phi = d_inv_sqrt[:, None] * vecs
        # drop the trivial (constant) leading eigenvector
        self.eigenvalues_ = vals[1:self.n_components + 1]
        self.eigenvectors_ = phi[:, 1:self.n_components + 1]
        self.embedding_ = self.eigenvectors_ * (self.eigenvalues_ ** self.t)
        return self

    def transform(self, Y):
        Y = np.asarray(Y, dtype=float)
        if np.array_equal(Y, self.X_fit_):
            return self.embedding_
        D2 = cdist(Y, self.X_fit_, "sqeuclidean")
        K = np.exp(-D2 / self.epsilon_)
        qy = K.sum(axis=1)
        Ka = K / np.outer(qy ** self.alpha, self.q_fit_ ** self.alpha)
        dy = Ka.sum(axis=1)
        P = Ka / dy[:, None]                       # row-stochastic kernel to landmarks
        # Nystrom: psi(y) = (1/lambda) * sum_j P(y, x_j) phi_j(x_j)
        psi = (P @ self.eigenvectors_) / self.eigenvalues_[None, :]
        return psi * (self.eigenvalues_ ** self.t)

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_


def trustworthiness(X_high, X_low, n_neighbors=10):
    """sklearn trustworthiness; higher = local neighborhoods better preserved."""
    n = X_high.shape[0]
    k = min(n_neighbors, max(1, (n - 1) // 2))
    return float(_trustworthiness(X_high, X_low, n_neighbors=k))


def geodesic_preservation(X_high, X_low, n_neighbors=10):
    """Spearman correlation between geodesic distances on the high-D kNN graph
    and Euclidean distances in the low-D embedding. Higher = global manifold
    geometry better preserved."""
    from scipy.stats import spearmanr

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_high)
    graph = nn.kneighbors_graph(mode="distance")
    geo = shortest_path(graph, method="D", directed=False)
    iu = np.triu_indices(X_high.shape[0], k=1)
    g = geo[iu]
    finite = np.isfinite(g)
    emb_d = squareform(pdist(X_low))[iu]
    rho, _ = spearmanr(g[finite], emb_d[finite])
    return float(rho)


def linear_reconstruction_r2(X_high, X_low):
    """R^2 of reconstructing the standardized high-D features from the 2D
    embedding via linear regression (multioutput, variance-weighted)."""
    lr = LinearRegression().fit(X_low, X_high)
    pred = lr.predict(X_low)
    return float(r2_score(X_high, pred, multioutput="variance_weighted"))
