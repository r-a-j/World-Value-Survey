"""Unit tests for Cumulative Link Ordered Logit, VIF, Proportional Odds, and LCA."""

from __future__ import annotations

import unittest
import numpy as np
from scipy.special import expit

from src.latent_class import fit_lca
from src.ordinal_model import (
    _nll_and_grad_natural,
    compute_vif,
    exact_hessian_natural,
    fit_ordered_logit,
    likelihood_ratio_test,
    proportional_odds_check,
)


def _simulate_ordinal(
    n: int,
    beta_true: np.ndarray,
    thresholds_true: np.ndarray,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = len(beta_true)
    X = rng.normal(size=(n, p))
    eta = X @ beta_true
    cum = expit(thresholds_true[None, :] - eta[:, None])
    u = rng.uniform(size=n)
    y = np.sum(u[:, None] > cum, axis=1).astype(np.int64)
    return X, y


class TestOrderedLogit(unittest.TestCase):
    def test_exact_analytic_gradient_and_hessian(self) -> None:
        beta_true = np.array([0.8, -0.5, 0.3])
        t_true = np.array([-1.2, 0.1, 1.4])
        X, y = _simulate_ordinal(400, beta_true, t_true, seed=7)

        beta_eval = np.array([0.5, -0.2, 0.1])
        t_eval = np.array([-1.0, 0.2, 1.3])
        nll0, gb, gt = _nll_and_grad_natural(beta_eval, t_eval, X, y, n_classes=4)
        grad_analytic = np.concatenate([gb, gt])
        H_analytic = exact_hessian_natural(beta_eval, t_eval, X, y, n_classes=4)

        theta0 = np.concatenate([beta_eval, t_eval])
        eps = 1e-6
        grad_fd = np.zeros_like(theta0)
        H_fd = np.zeros_like(H_analytic)
        for i in range(len(theta0)):
            step = np.zeros_like(theta0)
            step[i] = eps
            tp = theta0 + step
            tm = theta0 - step
            fp, gbp, gtp = _nll_and_grad_natural(tp[:3], tp[3:], X, y, n_classes=4)
            fm, gbm, gtm = _nll_and_grad_natural(tm[:3], tm[3:], X, y, n_classes=4)
            grad_fd[i] = (fp - fm) / (2.0 * eps)
            H_fd[:, i] = (np.concatenate([gbp, gtp]) - np.concatenate([gbm, gtm])) / (2.0 * eps)

        self.assertLess(np.max(np.abs(grad_analytic - grad_fd)), 1e-6)
        self.assertLess(np.max(np.abs(H_analytic - H_fd)), 1e-5)

    def test_parameter_recovery_lrt_and_proportional_odds(self) -> None:
        beta_true = np.array([0.9, -0.6, 0.0])
        t_true = np.array([-1.3, 0.0, 1.3])
        X, y = _simulate_ordinal(5000, beta_true, t_true, seed=42)

        res_full = fit_ordered_logit(X, y, ["x1", "x2", "x3"], n_classes=4)
        self.assertTrue(res_full.converged)
        self.assertLess(np.max(np.abs(res_full.beta - beta_true) / res_full.se_beta), 3.0)
        self.assertLess(np.max(np.abs(res_full.thresholds - t_true) / res_full.se_thresholds), 3.0)
        self.assertLess(np.max(np.abs(res_full.beta - beta_true)), 0.10)
        self.assertLess(np.max(np.abs(res_full.thresholds - t_true)), 0.12)
        self.assertGreater(res_full.mcfadden_r2, 0.08)

        # Predict high trust probability with delta-method CI
        x_zero = np.zeros((1, 3))
        p_hi, lo, hi = res_full.predict_high_trust_with_ci(x_zero, threshold_idx=1)
        # True P(Y >= 2 | x=0) = 1 - sigmoid(0) = 0.5
        self.assertAlmostEqual(float(p_hi[0]), 0.5, delta=0.03)
        self.assertLess(float(lo[0]), float(p_hi[0]))
        self.assertGreater(float(hi[0]), float(p_hi[0]))

        # LRT removing strong predictor x2 must be highly significant
        res_rest = fit_ordered_logit(X[:, [0, 2]], y, ["x1", "x3"], n_classes=4)
        stat, df, p_val = likelihood_ratio_test(res_rest, res_full)
        self.assertEqual(df, 1)
        self.assertGreater(stat, 200.0)
        self.assertLess(p_val, 1e-20)

        # Proportional odds check under true PO data -> small spread across cutpoints
        po = proportional_odds_check(X, y, ["x1", "x2", "x3"], n_classes=4)
        self.assertLess(float(po[0]["max_beta_spread"]), 0.20)
        self.assertLess(float(po[1]["max_beta_spread"]), 0.20)

    def test_vif_exact_and_always_ge_one(self) -> None:
        rng = np.random.default_rng(99)
        n = 4000
        x1 = rng.normal(size=n)
        # x2 has correlation ~ 0.8 with x1 -> theoretical VIF = 1 / (1 - 0.64) = 2.7778
        x2 = 0.8 * x1 + 0.6 * rng.normal(size=n)
        x3 = rng.normal(size=n)
        X = np.column_stack([x1, x2, x3])
        vifs = {row["feature"]: float(row["vif"]) for row in compute_vif(X, ["x1", "x2", "x3"])}
        for val in vifs.values():
            self.assertGreaterEqual(val, 1.0)
        self.assertAlmostEqual(vifs["x1"], 1.0 / (1.0 - 0.64), delta=0.15)
        self.assertAlmostEqual(vifs["x2"], 1.0 / (1.0 - 0.64), delta=0.15)
        self.assertAlmostEqual(vifs["x3"], 1.0, delta=0.05)


class TestLatentClassAnalysis(unittest.TestCase):
    def test_lca_recovery_and_bic_selection(self) -> None:
        rng = np.random.default_rng(2025)
        n = 3000
        # 3 clearly distinct discrete 4-level classes across 6 items
        true_pi = np.array([0.30, 0.45, 0.25])
        true_theta = np.zeros((3, 6, 4), dtype=np.float64)
        # Class 0: low trust across all 6
        true_theta[0, :, :] = [0.55, 0.30, 0.10, 0.05]
        # Class 1: selective (high on items 0..2, low on items 3..5)
        true_theta[1, :3, :] = [0.05, 0.15, 0.50, 0.30]
        true_theta[1, 3:, :] = [0.40, 0.40, 0.15, 0.05]
        # Class 2: broad high trust
        true_theta[2, :, :] = [0.04, 0.08, 0.48, 0.40]

        z = rng.choice(3, size=n, p=true_pi)
        Y = np.zeros((n, 6), dtype=np.int64)
        for i in range(n):
            for j in range(6):
                Y[i, j] = rng.choice(4, p=true_theta[z[i], j])

        names = [f"item_{j}" for j in range(6)]
        res2 = fit_lca(Y, names, n_classes=2, n_init=8, seed=1)
        res3 = fit_lca(Y, names, n_classes=3, n_init=8, seed=1)
        res4 = fit_lca(Y, names, n_classes=4, n_init=8, seed=1)

        # EM log-likelihood must be monotonically non-decreasing (up to floating noise)
        diffs = np.diff(res3.ll_history)
        self.assertTrue(np.all(diffs >= -1e-6))

        # True K=3 must beat both K=2 and K=4 on BIC
        self.assertLess(res3.bic, res2.bic)
        self.assertLess(res3.bic, res4.bic)
        self.assertGreater(res3.entropy, 0.72)

        # Conditional high-trust probabilities match ground truth closely
        true_hi = true_theta[:, :, 2:].sum(axis=2)
        est_hi = res3.high_trust_matrix()
        self.assertLess(np.max(np.abs(est_hi - true_hi)), 0.05)


if __name__ == "__main__":
    unittest.main()
