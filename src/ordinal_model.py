"""Exact Cumulative Link Ordinal Logistic Regression, Proportional Odds Checks, and VIF.

All confidence items are coded on an ascending ordinal scale so higher values mean
higher institutional trust:
    0 -> 1 (None at all)
    1 -> 2 (Not very much)
    2 -> 3 (Quite a lot)
    3 -> 4 (A great deal)

Under the standard cumulative logit parameterization (matching MASS::polr and
ordinal::clm):
    P(Y_i <= j | x_i) = sigmoid(alpha_j - x_i^T beta),   alpha_0 < alpha_1 < ... < alpha_{K-2}

A positive coefficient beta_p > 0 strictly increases the cumulative odds of higher
trust P(Y > j) / P(Y <= j) by a multiplicative factor exp(beta_p).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm


@dataclass
class OrderedLogitResult:
    """Fitted Cumulative Link Ordinal Logistic Regression result."""

    feature_names: list[str]
    n_obs: int
    n_classes: int
    beta: np.ndarray
    thresholds: np.ndarray
    se_beta: np.ndarray
    se_thresholds: np.ndarray
    z_beta: np.ndarray
    p_beta: np.ndarray
    odds_ratio: np.ndarray
    or_ci_low: np.ndarray
    or_ci_high: np.ndarray
    log_likelihood: float
    null_log_likelihood: float
    mcfadden_r2: float
    aic: float
    bic: float
    cov: np.ndarray
    converged: bool

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """Predict K-class probabilities (N, K) for design matrix X."""
        X = np.asarray(X, dtype=np.float64)
        eta = X @ self.beta
        cum = expit(self.thresholds[None, :] - eta[:, None])
        probs = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        probs[:, 0] = cum[:, 0]
        for j in range(1, self.n_classes - 1):
            probs[:, j] = cum[:, j] - cum[:, j - 1]
        probs[:, -1] = 1.0 - cum[:, -1]
        return np.clip(probs, 0.0, 1.0)

    def predict_high_trust_with_ci(
        self, X: np.ndarray, threshold_idx: int = 1, alpha_level: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict P(Y > threshold_idx | X) with exact delta-method confidence intervals.

        For a 4-level scale (0=None, 1=Not very much, 2=Quite a lot, 3=A great deal),
        threshold_idx=1 returns P(Y >= 2), i.e. P(Quite a lot or A great deal).
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        m = self.n_classes - 1
        # Linear predictor for P(Y > j) = sigmoid(eta - alpha_j)
        lp = X @ self.beta - self.thresholds[threshold_idx]
        # Gradient of lp wrt [beta (p), thresholds (m)]
        grad = np.zeros((n, p + m), dtype=np.float64)
        grad[:, :p] = X
        grad[:, p + threshold_idx] = -1.0
        var_lp = np.einsum("ni,ij,nj->n", grad, self.cov, grad)
        se_lp = np.sqrt(np.maximum(var_lp, 0.0))
        z_crit = norm.ppf(1.0 - alpha_level / 2.0)
        mean_p = expit(lp)
        ci_low = expit(lp - z_crit * se_lp)
        ci_high = expit(lp + z_crit * se_lp)
        return mean_p, ci_low, ci_high


def _unpack_unconstrained(params: np.ndarray, p: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Map unconstrained optimizer vector to (beta, strictly increasing thresholds)."""
    beta = params[:p]
    raw_t = params[p : p + m]
    thresholds = np.empty(m, dtype=np.float64)
    thresholds[0] = raw_t[0]
    for j in range(1, m):
        thresholds[j] = thresholds[j - 1] + np.exp(np.clip(raw_t[j], -20.0, 20.0))
    return beta, thresholds


def _nll_and_grad_natural(
    beta: np.ndarray,
    thresholds: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    l2_reg: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Negative log-likelihood and exact gradient in natural (beta, thresholds) coordinates."""
    n, p = X.shape
    m = n_classes - 1
    eta = X @ beta

    has_u = y < m
    has_l = y > 0

    F_u = np.ones(n, dtype=np.float64)
    F_l = np.zeros(n, dtype=np.float64)
    if np.any(has_u):
        F_u[has_u] = expit(thresholds[y[has_u]] - eta[has_u])
    if np.any(has_l):
        F_l[has_l] = expit(thresholds[y[has_l] - 1] - eta[has_l])

    probs = np.clip(F_u - F_l, 1e-15, 1.0)
    nll = -float(np.sum(np.log(probs))) + 0.5 * l2_reg * float(np.dot(beta, beta))

    g_u = np.where(has_u, F_u * (1.0 - F_u), 0.0)
    g_l = np.where(has_l, F_l * (1.0 - F_l), 0.0)

    # d(-ell)/d(eta) = -(g_l - g_u) / probs = (g_u - g_l) / probs
    d_nll_d_eta = (g_u - g_l) / probs
    grad_beta = X.T @ d_nll_d_eta + l2_reg * beta

    grad_t = np.zeros(m, dtype=np.float64)
    for j in range(m):
        mask_u = y == j
        mask_l = y == (j + 1)
        val = 0.0
        if np.any(mask_u):
            val -= float(np.sum(g_u[mask_u] / probs[mask_u]))
        if np.any(mask_l):
            val += float(np.sum(g_l[mask_l] / probs[mask_l]))
        grad_t[j] = val

    return nll, grad_beta, grad_t


def exact_hessian_natural(
    beta: np.ndarray,
    thresholds: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    l2_reg: float = 0.0,
) -> np.ndarray:
    """Exact analytic Hessian of negative log-likelihood in natural (beta, thresholds) coordinates."""
    n, p = X.shape
    m = n_classes - 1
    eta = X @ beta

    has_u = y < m
    has_l = y > 0

    F_u = np.ones(n, dtype=np.float64)
    F_l = np.zeros(n, dtype=np.float64)
    if np.any(has_u):
        F_u[has_u] = expit(thresholds[y[has_u]] - eta[has_u])
    if np.any(has_l):
        F_l[has_l] = expit(thresholds[y[has_l] - 1] - eta[has_l])

    probs = np.clip(F_u - F_l, 1e-15, 1.0)
    g_u = np.where(has_u, F_u * (1.0 - F_u), 0.0)
    g_l = np.where(has_l, F_l * (1.0 - F_l), 0.0)
    h_u = g_u * (1.0 - 2.0 * F_u)
    h_l = g_l * (1.0 - 2.0 * F_l)

    H = np.zeros((p + m, p + m), dtype=np.float64)

    # Beta-Beta block
    w_ee = -(h_u - h_l) / probs + ((g_l - g_u) / probs) ** 2
    H[:p, :p] = (X.T * w_ee) @ X
    if l2_reg > 0.0:
        H[:p, :p] += l2_reg * np.eye(p)

    # Beta-Threshold and Threshold-Threshold blocks
    for j in range(m):
        mask_u = y == j
        mask_l = y == (j + 1)

        w_et = np.zeros(n, dtype=np.float64)
        if np.any(mask_u):
            w_et[mask_u] = h_u[mask_u] / probs[mask_u] + (
                (g_l[mask_u] - g_u[mask_u]) * g_u[mask_u]
            ) / (probs[mask_u] ** 2)
        if np.any(mask_l):
            w_et[mask_l] = -h_l[mask_l] / probs[mask_l] - (
                (g_l[mask_l] - g_u[mask_l]) * g_l[mask_l]
            ) / (probs[mask_l] ** 2)

        col = X.T @ w_et
        H[:p, p + j] = col
        H[p + j, :p] = col

        # Diagonal threshold entry
        diag_val = 0.0
        if np.any(mask_u):
            diag_val += float(
                np.sum(-h_u[mask_u] / probs[mask_u] + (g_u[mask_u] / probs[mask_u]) ** 2)
            )
        if np.any(mask_l):
            diag_val += float(
                np.sum(h_l[mask_l] / probs[mask_l] + (g_l[mask_l] / probs[mask_l]) ** 2)
            )
        H[p + j, p + j] = diag_val

        # Adjacent off-diagonal (j, j-1): occurs where y == j
        if j > 0 and np.any(mask_u):
            off_val = float(np.sum(-(g_u[mask_u] * g_l[mask_u]) / (probs[mask_u] ** 2)))
            H[p + j, p + j - 1] = off_val
            H[p + j - 1, p + j] = off_val

    return H


def fit_ordered_logit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_classes: int = 4,
    l2_reg: float = 0.0,
) -> OrderedLogitResult:
    """Fit exact Cumulative Link Ordinal Logistic Regression via L-BFGS-B.

    Standardizes predictors internally during optimization for clean conditioning,
    then transforms coefficients and the exact Hessian back to the original feature scale.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, p = X.shape
    m = n_classes - 1
    if len(feature_names) != p:
        raise ValueError("feature_names length must match X.shape[1]")
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError(f"y must contain integers in [0, {n_classes - 1}]")

    # Standardize columns internally for fast, rock-solid L-BFGS-B convergence
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds_safe = np.where(stds > 1e-12, stds, 1.0)
    Xs = (X - means) / stds_safe

    # Empirical cumulative logits for initial thresholds
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    cum_p = np.clip(np.cumsum(counts[:-1]) / n, 1e-4, 1.0 - 1e-4)
    t_init = np.log(cum_p / (1.0 - cum_p))
    raw_t_init = np.empty(m, dtype=np.float64)
    raw_t_init[0] = t_init[0]
    for j in range(1, m):
        raw_t_init[j] = np.log(max(t_init[j] - t_init[j - 1], 1e-2))

    # Null model log-likelihood (intercept-only ordinal distribution)
    p_emp = np.clip(counts / n, 1e-15, 1.0)
    null_ll = float(np.sum(counts * np.log(p_emp)))

    def obj_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        beta_s, t_s = _unpack_unconstrained(params, p, m)
        nll, gb, gt = _nll_and_grad_natural(beta_s, t_s, Xs, y, n_classes, l2_reg=0.0)
        if l2_reg > 0.0:
            # Penalty on original scale beta = beta_s / stds_safe
            beta_orig = beta_s / stds_safe
            nll += 0.5 * l2_reg * float(np.dot(beta_orig, beta_orig))
            gb = gb + l2_reg * beta_orig / stds_safe

        grad = np.empty_like(params)
        grad[:p] = gb
        raw_t = params[p : p + m]
        # Chain rule from thresholds to unconstrained raw_t
        suffix_sum = np.cumsum(gt[::-1])[::-1]
        grad[p] = suffix_sum[0]
        for j in range(1, m):
            grad[p + j] = np.exp(np.clip(raw_t[j], -20.0, 20.0)) * suffix_sum[j]
        return nll, grad

    x0 = np.concatenate([np.zeros(p, dtype=np.float64), raw_t_init])
    opt = minimize(
        obj_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
    )

    beta_s, t_s = _unpack_unconstrained(opt.x, p, m)
    # Transform back to original unstandardized predictor scale:
    # t_s - ((X - mu)/s) @ beta_s = (t_s + (mu/s) @ beta_s) - X @ (beta_s / s)
    beta = beta_s / stds_safe
    thresholds = t_s + float(np.dot(means, beta))

    # Exact log-likelihood and analytic Hessian on the original scale
    nll_orig, _, _ = _nll_and_grad_natural(beta, thresholds, X, y, n_classes, l2_reg=0.0)
    ll = -nll_orig
    H = exact_hessian_natural(beta, thresholds, X, y, n_classes, l2_reg=l2_reg)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    se_all = np.sqrt(np.maximum(np.diag(cov), 0.0))
    se_beta = se_all[:p]
    se_thresholds = se_all[p:]

    z_beta = np.divide(beta, se_beta, out=np.zeros_like(beta), where=se_beta > 0)
    p_beta = 2.0 * norm.sf(np.abs(z_beta))

    z_crit = norm.ppf(0.975)
    odds_ratio = np.exp(beta)
    or_ci_low = np.exp(beta - z_crit * se_beta)
    or_ci_high = np.exp(beta + z_crit * se_beta)

    k_params = p + m
    aic = 2.0 * k_params - 2.0 * ll
    bic = k_params * np.log(n) - 2.0 * ll
    mcfadden_r2 = 1.0 - (ll / null_ll) if null_ll < 0 else 0.0

    return OrderedLogitResult(
        feature_names=list(feature_names),
        n_obs=n,
        n_classes=n_classes,
        beta=beta,
        thresholds=thresholds,
        se_beta=se_beta,
        se_thresholds=se_thresholds,
        z_beta=z_beta,
        p_beta=p_beta,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        mcfadden_r2=mcfadden_r2,
        aic=aic,
        bic=bic,
        cov=cov,
        converged=bool(opt.success),
    )


def likelihood_ratio_test(
    restricted: OrderedLogitResult, full: OrderedLogitResult
) -> tuple[float, int, float]:
    """Exact Likelihood Ratio Test comparing two nested models on the same dataset."""
    if restricted.n_obs != full.n_obs:
        raise ValueError(
            f"LRT requires identical sample size (got {restricted.n_obs} vs {full.n_obs})"
        )
    df = (len(full.beta) + len(full.thresholds)) - (
        len(restricted.beta) + len(restricted.thresholds)
    )
    if df <= 0:
        raise ValueError(f"Full model must have strictly more parameters (df={df})")
    stat = max(0.0, 2.0 * (full.log_likelihood - restricted.log_likelihood))
    p_val = float(chi2.sf(stat, df))
    return float(stat), int(df), p_val


def _fit_binary_logit(X: np.ndarray, y_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact binary logistic regression returning (beta_slopes, se_slopes) excluding intercept."""
    n, p = X.shape
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds_safe = np.where(stds > 1e-12, stds, 1.0)
    Xs = (X - means) / stds_safe
    Xd = np.column_stack([np.ones(n, dtype=np.float64), Xs])

    def nll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = Xd @ theta
        prob = expit(z)
        # Numerically stable binary cross-entropy
        nll = float(np.sum(np.logaddexp(0.0, z) - y_bin * z))
        grad = Xd.T @ (prob - y_bin)
        return nll, grad

    opt = minimize(
        nll_grad,
        np.zeros(p + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
    )
    theta_s = opt.x
    beta = theta_s[1:] / stds_safe
    intercept = theta_s[0] - float(np.dot(means, beta))

    X_orig = np.column_stack([np.ones(n, dtype=np.float64), X])
    prob_orig = expit(X_orig @ np.concatenate([[intercept], beta]))
    w = prob_orig * (1.0 - prob_orig)
    H = (X_orig.T * w) @ X_orig
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se_beta = np.sqrt(np.maximum(np.diag(cov)[1:], 0.0))
    return beta, se_beta


def proportional_odds_check(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_classes: int = 4,
) -> list[dict[str, float | str]]:
    """Evaluate the Proportional Odds assumption by fitting K-1 cumulative binary logits.

    Fits separate binary logits for Y >= 1, Y >= 2, ..., Y >= K-1 and reports each
    slope alongside the max spread across thresholds, allowing direct inspection of
    which predictors hold parallel odds and which show threshold-dependent slopes.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    m = n_classes - 1
    betas = np.zeros((m, X.shape[1]), dtype=np.float64)
    ses = np.zeros((m, X.shape[1]), dtype=np.float64)
    for cut in range(m):
        y_bin = (y >= (cut + 1)).astype(np.float64)
        betas[cut], ses[cut] = _fit_binary_logit(X, y_bin)

    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(feature_names):
        entry: dict[str, float | str] = {"feature": str(name)}
        for cut in range(m):
            entry[f"beta_ge_{cut + 2}"] = float(betas[cut, idx])
            entry[f"se_ge_{cut + 2}"] = float(ses[cut, idx])
            entry[f"or_ge_{cut + 2}"] = float(np.exp(betas[cut, idx]))
        entry["max_beta_spread"] = float(np.max(betas[:, idx]) - np.min(betas[:, idx]))
        rows.append(entry)
    return rows


def compute_vif(X: np.ndarray, feature_names: Sequence[str]) -> list[dict[str, float | str]]:
    """Compute exact Variance Inflation Factors VIF_k = 1 / (1 - R_k^2) >= 1.0.

    Always includes an explicit intercept column in every auxiliary OLS regression so
    R_k^2 lies strictly in [0, 1) and VIF_k is mathematically guaranteed >= 1.0.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if len(feature_names) != p:
        raise ValueError("feature_names length must match X.shape[1]")
    ones = np.ones((n, 1), dtype=np.float64)
    results: list[dict[str, float | str]] = []
    for k in range(p):
        yk = X[:, k]
        other_idx = [j for j in range(p) if j != k]
        if not other_idx:
            results.append({"feature": str(feature_names[k]), "r2": 0.0, "vif": 1.0})
            continue
        Xk = np.hstack([ones, X[:, other_idx]])
        coef, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
        pred = Xk @ coef
        ss_res = float(np.sum((yk - pred) ** 2))
        ss_tot = float(np.sum((yk - np.mean(yk)) ** 2))
        r2 = 0.0 if ss_tot <= 1e-15 else max(0.0, min(1.0 - ss_res / ss_tot, 1.0 - 1e-12))
        vif = 1.0 / (1.0 - r2)
        results.append({"feature": str(feature_names[k]), "r2": r2, "vif": vif})
    return results
