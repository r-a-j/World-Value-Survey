"""Model-Based Latent Class Analysis (LCA) for Discrete Ordinal Survey Data.

Implements a multi-start Expectation-Maximization (EM) finite categorical mixture
model, replacing Euclidean K-means on discrete 4-level institutional trust items.

Model:
    P(y_i) = sum_{k=1}^{K} pi_k * prod_{j=1}^{J} theta_{k, j, y_{ij}}

where y_{ij} in {0, 1, ..., M-1} is respondent i's discrete ordinal trust category
for institution j (0=None at all, 1=Not very much, 2=Quite a lot, 3=A great deal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp


@dataclass
class LCAResult:
    """Fitted Latent Class Analysis model."""

    n_classes: int
    n_obs: int
    n_items: int
    n_categories: int
    item_names: list[str]
    class_weights: np.ndarray          # (K,)
    item_probs: np.ndarray             # (K, J, M)
    responsibilities: np.ndarray       # (N, K)
    hard_assignments: np.ndarray       # (N,)
    log_likelihood: float
    n_params: int
    aic: float
    bic: float
    sabic: float
    entropy: float
    n_iter: int
    converged: bool
    ll_history: list[float]

    def high_trust_matrix(self, high_cats: tuple[int, ...] = (2, 3)) -> np.ndarray:
        """Return (K, J) matrix of conditional P(Y_j in high_cats | Class k)."""
        return np.sum(self.item_probs[:, :, list(high_cats)], axis=2)

    def mean_score_matrix(self) -> np.ndarray:
        """Return (K, J) matrix of expected 1..M ordinal trust score per class & item."""
        levels = np.arange(1, self.n_categories + 1, dtype=np.float64)
        return np.einsum("kjm,m->kj", self.item_probs, levels)


def _fit_single_em(
    Y: np.ndarray,
    n_classes: int,
    n_categories: int,
    rng: np.random.Generator,
    max_iter: int = 300,
    tol: float = 1e-6,
    smoothing: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float], bool]:
    """Run one EM start for a categorical finite mixture model."""
    n, j_items = Y.shape
    m = n_categories

    # Precompute one-hot indicator matrices for fast vectorized M-steps: (J, M, N)
    # Using flattened item-category indicator matrix (N, J * M) -> single BLAS matmul!
    offsets = (np.arange(j_items) * m)[None, :]
    flat_idx = Y + offsets
    indicator = np.zeros((n, j_items * m), dtype=np.float64)
    indicator[np.arange(n)[:, None], flat_idx] = 1.0

    # Random Dirichlet initialization for class weights and conditional probabilities
    pi = rng.dirichlet(np.full(n_classes, 2.0))
    theta = rng.dirichlet(np.full(m, 1.5), size=(n_classes, j_items))  # (K, J, M)

    ll_history: list[float] = []
    converged = False
    prev_ll = -np.inf

    for _ in range(max_iter):
        # E-step: log P(y_i, z_i = k) = log pi_k + sum_j log theta_{k, j, y_ij}
        log_theta_flat = np.log(np.clip(theta.reshape(n_classes, j_items * m), 1e-12, 1.0))
        log_joint = indicator @ log_theta_flat.T + np.log(np.clip(pi, 1e-12, 1.0))[None, :]
        log_norm = logsumexp(log_joint, axis=1)
        ll = float(np.sum(log_norm))
        ll_history.append(ll)

        log_resp = log_joint - log_norm[:, None]
        resp = np.exp(log_resp)

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # M-step with tiny Laplace smoothing for numerical stability
        nk = resp.sum(axis=0)
        pi = (nk + smoothing) / (n + n_classes * smoothing)

        counts = (resp.T @ indicator).reshape(n_classes, j_items, m)
        theta = (counts + smoothing) / (
            counts.sum(axis=2, keepdims=True) + m * smoothing
        )

    return pi, theta, resp, ll_history[-1], ll_history, converged


def fit_lca(
    Y: np.ndarray,
    item_names: Sequence[str],
    n_classes: int,
    n_categories: int = 4,
    n_init: int = 12,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
) -> LCAResult:
    """Fit a multi-start Latent Class Analysis model and sort classes canonically.

    Classes are ordered from lowest to highest overall mean institutional trust so
    class labels are invariant to EM permutation symmetry.
    """
    Y = np.asarray(Y, dtype=np.int64)
    n, j_items = Y.shape
    if len(item_names) != j_items:
        raise ValueError("item_names length must match Y.shape[1]")
    if np.any((Y < 0) | (Y >= n_categories)):
        raise ValueError(f"Y must contain integers in [0, {n_categories - 1}]")

    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float], bool] | None = None
    for _ in range(n_init):
        cand = _fit_single_em(
            Y,
            n_classes=n_classes,
            n_categories=n_categories,
            rng=rng,
            max_iter=max_iter,
            tol=tol,
        )
        if best is None or cand[3] > best[3]:
            best = cand

    assert best is not None
    pi, theta, resp, ll, ll_hist, converged = best

    # Canonical sort: order classes by average expected score across all J items
    levels = np.arange(1, n_categories + 1, dtype=np.float64)
    mean_trust_per_class = np.einsum("kjm,m->k", theta, levels) / j_items
    order = np.argsort(mean_trust_per_class)
    pi = pi[order]
    theta = theta[order]
    resp = resp[:, order]
    hard = np.argmax(resp, axis=1)

    # Free parameter count: (K - 1) weights + K * J * (M - 1) conditional probabilities
    n_params = (n_classes - 1) + n_classes * j_items * (n_categories - 1)
    aic = 2.0 * n_params - 2.0 * ll
    bic = n_params * np.log(n) - 2.0 * ll
    sabic = n_params * np.log((n + 2.0) / 24.0) - 2.0 * ll

    # Normalized classification entropy E_K in [0, 1] (higher = sharper separation)
    if n_classes > 1:
        r_safe = np.clip(resp, 1e-15, 1.0)
        raw_ent = -float(np.sum(r_safe * np.log(r_safe)))
        entropy = max(0.0, min(1.0, 1.0 - raw_ent / (n * np.log(n_classes))))
    else:
        entropy = 1.0

    return LCAResult(
        n_classes=n_classes,
        n_obs=n,
        n_items=j_items,
        n_categories=n_categories,
        item_names=list(item_names),
        class_weights=pi,
        item_probs=theta,
        responsibilities=resp,
        hard_assignments=hard,
        log_likelihood=ll,
        n_params=n_params,
        aic=aic,
        bic=bic,
        sabic=sabic,
        entropy=entropy,
        n_iter=len(ll_hist),
        converged=converged,
        ll_history=ll_hist,
    )
