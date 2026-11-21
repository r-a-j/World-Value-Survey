"""Generate large-font, single-panel standalone figures from exported WVS results CSVs.

Each chart uses a focused ~4:3 single-panel canvas with 11-13 pt typography so every
label, percentage, odds ratio, and confidence interval stays readable at card scale
without zooming.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.2,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 9.8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
        }
    )


def _save(fig: plt.Figure, png_path: Path, webp_dir: Path | None, webp_name: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    if webp_dir is not None:
        webp_dir.mkdir(parents=True, exist_ok=True)
        Image.open(png_path).convert("RGB").save(
            webp_dir / webp_name, format="WEBP", quality=92, method=6
        )


def build_all_panels(results_dir: Path, png_dir: Path, webp_dir: Path | None = None) -> None:
    _apply_style()

    df_g8 = pd.read_csv(results_dir / "g8_country_distribution.csv")
    df_macro = pd.read_csv(results_dir / "global_66_macro_trust.csv")
    df_usa = pd.read_csv(results_dir / "usa_ordered_logit.csv")
    df_po = pd.read_csv(results_dir / "usa_proportional_odds.csv").set_index("feature")
    df_joint = pd.read_csv(results_dir / "g8_joint_model.csv")
    df_u_ts = pd.read_csv(results_dir / "longitudinal_usa_institutions.csv")
    df_g_ts = pd.read_csv(results_dir / "longitudinal_g8_government.csv")
    df_lca_p = pd.read_csv(results_dir / "lca_item_profiles.csv")
    df_lca_s = pd.read_csv(results_dir / "lca_country_shares.csv")

    # 1. G8 4-level Government Trust Distribution (wide crisp hero)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=220)
    d1 = df_g8.sort_values("high_trust_pct", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(d1))
    colors = ["#c0392b", "#e67e22", "#3b82f6", "#1e3a8a"]
    cols = ["p_lvl_1", "p_lvl_2", "p_lvl_3", "p_lvl_4"]
    labels = ["None at all (1)", "Not very much (2)", "Quite a lot (3)", "A great deal (4)"]
    left = np.zeros(len(d1))
    for col, c, lbl in zip(cols, colors, labels):
        vals = 100.0 * d1[col].to_numpy()
        ax.barh(y_pos, vals, left=left, height=0.60, color=c, edgecolor="white", linewidth=1.0, label=lbl)
        for i, v in enumerate(vals):
            if v >= 6.5:
                txt_col = "white" if c in ("#c0392b", "#1e3a8a", "#3b82f6") else "#1e293b"
                ax.text(left[i] + v / 2.0, y_pos[i], f"{v:.1f}%", ha="center", va="center", fontsize=10.0, fontweight="bold", color=txt_col)
        left += vals
    for i, r in d1.iterrows():
        ax.text(101.6, i, f"High: {r['high_trust_pct']:.1f}%", va="center", ha="left", fontsize=10.6, fontweight="bold", color="#0f172a")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['country_name']} ({int(r['year'])}, n={int(r['n_gov_valid']):,})" for _, r in d1.iterrows()], fontsize=10.4, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of Valid Respondents (%)")
    ax.set_title("Government Trust Distribution Across 6 G8 Wave 7 Nations (Ascending 1-4 Scale)", loc="left", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=4, frameon=False)
    _save(fig, png_dir / "g8_gov_trust_distribution.png", webp_dir, "g8-gov-trust-distribution.webp")

    # 2. Critical Citizens Paradox: V-Dem Press Freedom vs Government Trust
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=220)
    pm = df_macro[["country", "is_g8", "VDem_PressFree", "gov_high_pct"]].dropna()
    other = pm[~pm["is_g8"]]
    g8p = pm[pm["is_g8"]]
    ax.scatter(other["VDem_PressFree"], other["gov_high_pct"], s=52, color="#94a3b8", alpha=0.68, edgecolors="white", linewidth=0.6, label=f"Other WVS nations (n={len(other)})")
    ax.scatter(g8p["VDem_PressFree"], g8p["gov_high_pct"], s=110, color="#dc2626", edgecolors="#0f172a", linewidth=1.1, zorder=5, label="G8 Wave 7 nations (n=6)")
    sl, ic = np.polyfit(pm["VDem_PressFree"], pm["gov_high_pct"], 1)
    xs = np.linspace(pm["VDem_PressFree"].min(), pm["VDem_PressFree"].max(), 100)
    ax.plot(xs, sl * xs + ic, color="#dc2626", linestyle="--", linewidth=2.0, alpha=0.88)
    r_v, p_v = pearsonr(pm["VDem_PressFree"], pm["gov_high_pct"])
    offs = {"CAN": (-38, 14), "DEU": (-10, 15), "JPN": (-40, 4), "GBR": (-38, -8), "USA": (-38, -12), "RUS": (10, 7)}
    for _, r in g8p.iterrows():
        dx, dy = offs.get(r["country"], (8, 5))
        ax.annotate(r["country"], (r["VDem_PressFree"], r["gov_high_pct"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=10.2, fontweight="bold", color="#0f172a", arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.8))
    ax.set_title(f"Critical Citizens: Press Freedom vs. Gov. Trust\n(n = {len(pm)} nations: Pearson r = {r_v:+.2f}, p = {p_v:.2e})", loc="left", pad=10)
    ax.set_xlabel("V-Dem Press & Expression Freedom Index (0-1)")
    ax.set_ylabel("High Government Trust P(Y >= 3) (%)")
    ax.set_xlim(0.02, 1.04)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94)
    _save(fig, png_dir / "global_press_freedom_paradox.png", webp_dir, "global-press-freedom-paradox.webp")

    # 3. Rule of Law: Transparency International CPI vs Police Trust
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=220)
    pp = df_macro[["country", "is_g8", "CPI", "police_high_pct"]].dropna()
    other_p = pp[~pp["is_g8"]]
    g8_p = pp[pp["is_g8"]]
    ax.scatter(other_p["CPI"], other_p["police_high_pct"], s=52, color="#94a3b8", alpha=0.68, edgecolors="white", linewidth=0.6, label=f"Other WVS nations (n={len(other_p)})")
    ax.scatter(g8_p["CPI"], g8_p["police_high_pct"], s=110, color="#1d4ed8", edgecolors="#0f172a", linewidth=1.1, zorder=5, label="G8 Wave 7 nations (n=6)")
    slp, icp = np.polyfit(pp["CPI"], pp["police_high_pct"], 1)
    xsp = np.linspace(pp["CPI"].min(), pp["CPI"].max(), 100)
    ax.plot(xsp, slp * xsp + icp, color="#1d4ed8", linestyle="--", linewidth=2.0, alpha=0.88)
    r_p, p_p = pearsonr(pp["CPI"], pp["police_high_pct"])
    offs_p = {"DEU": (9, 5), "CAN": (10, -7), "GBR": (-38, -14), "JPN": (-34, 8), "USA": (-38, -10), "RUS": (10, -8)}
    for _, r in g8_p.iterrows():
        dx, dy = offs_p.get(r["country"], (8, 5))
        ax.annotate(r["country"], (r["CPI"], r["police_high_pct"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=10.2, fontweight="bold", color="#0f172a", arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.8))
    ax.set_title(f"Rule of Law: Anti-Corruption (CPI) vs. Police Trust\n(n = {len(pp)} nations: Pearson r = {r_p:+.2f}, p = {p_p:.2e})", loc="left", pad=10)
    ax.set_xlabel("Transparency International CPI (0 = Corrupt, 100 = Clean)")
    ax.set_ylabel("High Police Trust P(Y >= 3) (%)")
    ax.set_ylim(10, 105)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94)
    _save(fig, png_dir / "global_cpi_police_trust.png", webp_dir, "global-cpi-police-trust.webp")

    # 4. USA Wave 7 Ordered Logit Odds Ratios + PO Cutpoint Check
    fig, ax = plt.subplots(figsize=(8.6, 6.6), dpi=220)
    uf = df_usa[(df_usa["model"] == "USA_Full_1B") & (~df_usa["term"].str.startswith("Threshold"))].sort_values("beta", ascending=True).reset_index(drop=True)
    inst_set = {"Political Parties", "Civil Services", "Elections", "Courts", "Police", "Armed Forces", "The Press"}
    pretty = {
        "Age_10yr": "Age (+10 years)",
        "Edu_Lower": "Education: Lower vs Middle",
        "Edu_Upper": "Education: Upper vs Middle",
        "Democracy Importance": "Democracy Importance (1-10)",
        "Favor Strong Leader": "Favor Strong Leader (1-4)",
        "Income Decile": "Income Decile (1-10)",
    }
    ax.axvline(1.0, color="#475569", linestyle="--", linewidth=1.2, zorder=1)
    for i, r in uf.iterrows():
        feat = r["term"]
        or_v, lo, hi = float(r["odds_ratio"]), float(r["or_ci_low"]), float(r["or_ci_high"])
        col = "#1d4ed8" if feat in inst_set else "#0f766e"
        ax.errorbar(or_v, i, xerr=[[or_v - lo], [hi - or_v]], fmt="o", color=col, ecolor=col, elinewidth=2.1, capsize=3.5, markersize=7.2, zorder=4)
        for cut_col, mk, mcol in [("or_ge_2", "^", "#f59e0b"), ("or_ge_3", "s", "#64748b"), ("or_ge_4", "v", "#dc2626")]:
            vc = float(df_po.loc[feat, cut_col])
            if 0.22 <= vc <= 5.5:
                ax.scatter(vc, i, marker=mk, s=28, facecolors="none", edgecolors=mcol, linewidths=1.1, zorder=5, alpha=0.88)
    press_idx = uf.index[uf["term"] == "The Press"][0]
    press_or = float(uf.loc[press_idx, "odds_ratio"])
    ax.annotate(f"2017 Press-vs-Gov\npolarization (OR={press_or:.2f})", xy=(press_or * 1.15, press_idx), xytext=(0.66, press_idx - 0.12),
                fontsize=9.6, fontweight="bold", color="#1e3a8a", va="center", arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    ax.scatter([], [], color="#1d4ed8", s=45, label="Institutional Trust OR (95% CI)")
    ax.scatter([], [], color="#0f766e", s=45, label="Socio-Political OR (95% CI)")
    ax.scatter([], [], marker="^", s=34, facecolors="none", edgecolors="#f59e0b", label="PO Cutpoint: Y >= 2")
    ax.scatter([], [], marker="s", s=34, facecolors="none", edgecolors="#64748b", label="PO Cutpoint: Y >= 3")
    ax.scatter([], [], marker="v", s=34, facecolors="none", edgecolors="#dc2626", label="PO Cutpoint: Y = 4")
    ax.set_xscale("log")
    ax.set_xlim(0.26, 4.8)
    ax.set_xticks([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.3", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.5"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(np.arange(len(uf)))
    ax.set_yticklabels([pretty.get(t, t) for t in uf["term"]], fontsize=10.0)
    ax.set_xlabel("Cumulative Odds Ratio for Higher Government Trust (Log Scale)")
    ax.set_title("USA Wave 7 Ordered Logit & Proportional Odds Check\n(n = 2,396 complete / 2,596 total; McFadden R² = 0.248)", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)
    _save(fig, png_dir / "usa_ordered_logit_forest.png", webp_dir, "usa-ordered-logit-forest.webp")

    # 5. Joint 6-Country G8 Categorical Country Fixed Effects
    fig, ax = plt.subplots(figsize=(8.2, 5.9), dpi=220)
    codes = ["RUS", "CAN", "JPN", "USA", "GBR"]
    cnames = {"RUS": "Russia", "CAN": "Canada", "JPN": "Japan", "USA": "United States", "GBR": "United Kingdom"}
    ms = df_joint[df_joint["model"] == "G8_Socio_CountryFE"].set_index("term")
    mf = df_joint[df_joint["model"] == "G8_Full_CountryFE_2B"].set_index("term")
    ax.axvline(1.0, color="#475569", linestyle="--", linewidth=1.3, label="Germany Reference (OR = 1.0)")
    for i, code in enumerate(codes):
        t = f"Country_{code}"
        ors, los, his = float(ms.loc[t, "odds_ratio"]), float(ms.loc[t, "or_ci_low"]), float(ms.loc[t, "or_ci_high"])
        ax.errorbar(ors, i + 0.14, xerr=[[ors - los], [his - ors]], fmt="s", color="#d97706", ecolor="#d97706",
                    elinewidth=2.1, capsize=4, markersize=7.2, label="Adjusted for Demographics & Attitudes" if i == 0 else None)
        ax.text(his * 1.06, i + 0.14, f"{ors:.2f}", va="center", ha="left", fontsize=10.0, color="#b45309", fontweight="bold")
        orf, lof, hif = float(mf.loc[t, "odds_ratio"]), float(mf.loc[t, "or_ci_low"]), float(mf.loc[t, "or_ci_high"])
        ax.errorbar(orf, i - 0.14, xerr=[[orf - lof], [hif - orf]], fmt="o", color="#1d4ed8", ecolor="#1d4ed8",
                    elinewidth=2.1, capsize=4, markersize=7.5, label="Adjusted for Other Inst. Trust + Demo." if i == 0 else None)
        ax.text(hif * 1.06, i - 0.14, f"{orf:.2f}", va="center", ha="left", fontsize=10.0, color="#1e3a8a", fontweight="bold")
    box_txt = "Pooled 6-G8: n = 11,371\nCountry FE LRT: χ²(5) = 647.2\nInteraction LRT: χ²(25) = 595.0\n(both p < 1e-100)\nR²: 0.245 (FE) -> 0.266 (Int.)"
    ax.text(0.98, 0.72, box_txt, transform=ax.transAxes, fontsize=9.2, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0))
    ax.set_xscale("log")
    ax.set_xlim(0.24, 6.2)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.25", "0.5", "1.0", "2.0", "4.0"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(np.arange(len(codes)))
    ax.set_yticklabels([cnames[c] for c in codes], fontsize=10.8, fontweight="bold")
    ax.set_ylim(-0.65, len(codes) - 0.25)
    ax.set_xlabel("Categorical Country Odds Ratio vs. Germany (Ref = 1.0)")
    ax.set_title("Joint 6-Country G8 Categorical Effects (Ref: Germany)\n(Pooled Ordered Logit with Nested Interaction LRT)", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.1)
    _save(fig, png_dir / "g8_country_fixed_effects.png", webp_dir, "g8-country-fixed-effects.webp")

    # 6. USA Longitudinal 6 Institutions (1995-2017)
    fig, ax = plt.subplots(figsize=(8.4, 6.1), dpi=220)
    inst_styles = {
        "Armed Forces": ("#1e3a8a", "o", "-"),
        "Police": ("#0284c7", "s", "-"),
        "Courts": ("#0f766e", "^", "--"),
        "Government": ("#d97706", "D", "-"),
        "Parliament": ("#dc2626", "v", "--"),
        "Political Parties": ("#7c3aed", "P", ":"),
    }
    for inst, (col, mk, ls) in inst_styles.items():
        sub = df_u_ts[df_u_ts["institution"] == inst].sort_values("year")
        ax.plot(sub["year"], sub["pred_high_pct"], color=col, marker=mk, linestyle=ls, linewidth=2.4, markersize=7.0, label=inst)
        ax.fill_between(sub["year"], sub["ci_low_pct"], sub["ci_high_pct"], color=col, alpha=0.14)
    ax.annotate("Armed Forces stays resilient\nat 81-85% across 22 years", xy=(2011, 84.5), xytext=(1998.0, 92.2),
                fontsize=9.6, fontweight="bold", color="#1e3a8a", arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.1))
    ax.annotate("Parties & Congress drop\nto 10-15% by 2017", xy=(2017, 10.4), xytext=(2003.2, 2.2),
                fontsize=9.6, fontweight="bold", color="#7c3aed", arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.1))
    ax.set_title("USA Institutional Trust Trajectories (1995-2017, n = 8,819)\n(Ordered Logit P(Y >= 3), 95% CI; Inst × Year LRT χ²(19) = 568)", loc="left", pad=10)
    ax.set_xlabel("WVS Survey Year")
    ax.set_ylabel("Predicted High Trust Probability P(Y >= 3) (%)")
    ax.set_xlim(1993.5, 2018.8)
    ax.set_ylim(0, 102)
    ax.set_xticks([1995, 1999, 2006, 2011, 2017])
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.14), ncol=3, frameon=False, fontsize=9.8)
    _save(fig, png_dir / "usa_longitudinal_institutions.png", webp_dir, "usa-longitudinal-institutions.webp")

    # 7. Cross-National G8 Government Trust Evolution (1995-2022)
    fig, ax = plt.subplots(figsize=(8.4, 6.1), dpi=220)
    c_styles = {
        "CAN": ("#0284c7", "o", "-", "Canada"),
        "DEU": ("#16a34a", "s", "-", "Germany"),
        "JPN": ("#9333ea", "^", "-", "Japan"),
        "RUS": ("#dc2626", "D", "--", "Russia"),
        "USA": ("#d97706", "v", "-", "United States"),
        "GBR": ("#334155", "P", "--", "United Kingdom"),
    }
    for code, (col, mk, ls, cname) in c_styles.items():
        sub = df_g_ts[df_g_ts["country"] == code].sort_values("year")
        ax.plot(sub["year"], sub["pred_high_pct"], color=col, marker=mk, linestyle=ls, linewidth=2.4, markersize=7.0, label=cname)
        ax.fill_between(sub["year"], sub["ci_low_pct"], sub["ci_high_pct"], color=col, alpha=0.13)
    ax.set_title("Cross-National G8 Government Trust (1995-2022)\n(Ordered Logit P(Y >= 3) at Age 45, Income 5, 95% CI)", loc="left", pad=10)
    ax.set_xlabel("WVS Survey Year")
    ax.set_ylabel("Predicted High Government Trust P(Y >= 3) (%)")
    ax.set_xlim(1994, 2023.5)
    ax.set_ylim(10, 68)
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.14), ncol=3, frameon=False, fontsize=9.8)
    _save(fig, png_dir / "g8_longitudinal_government.png", webp_dir, "g8-longitudinal-government.webp")

    # 8. LCA 4-Class Conditional Trust Profile Heatmap
    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=220)
    items = ["Armed Forces", "Police", "Courts", "Civil Services", "Elections", "Government", "Parliament", "Political Parties"]
    hi_mat = np.column_stack([df_lca_p[f"hi_{it}"].to_numpy() for it in items])
    gr_mat = np.column_stack([df_lca_p[f"great_{it}"].to_numpy() for it in items])
    im = ax.imshow(hi_mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for k in range(hi_mat.shape[0]):
        for j in range(hi_mat.shape[1]):
            v, g = hi_mat[k, j], gr_mat[k, j]
            tc = "white" if v >= 55 else "#0f172a"
            sc = "#e2e8f0" if v >= 55 else "#475569"
            ax.text(j, k - 0.13, f"{v:.0f}%", ha="center", va="center", fontsize=10.6, fontweight="bold", color=tc)
            ax.text(j, k + 0.22, f"({g:.0f}%)", ha="center", va="center", fontsize=8.4, color=sc)
    short_cls = [
        f"C1: Broad Distrustors\n({df_lca_p.loc[0, 'weight_pct']:.1f}% of G8)",
        f"C2: Rule-of-Law\nPragmatists ({df_lca_p.loc[1, 'weight_pct']:.1f}%)",
        f"C3: Moderate Trustors\n'Quite a lot' ({df_lca_p.loc[2, 'weight_pct']:.1f}%)",
        f"C4: Strong Trustors\n'A great deal' ({df_lca_p.loc[3, 'weight_pct']:.1f}%)",
    ]
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(short_cls, fontsize=9.8, fontweight="bold")
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels(items, rotation=26, ha="right", fontsize=10.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("Conditional P(High Trust >= 3) (%)", fontsize=10.0)
    ax.set_title("Latent Class Analysis (LCA) Trust Profiles (K = 4, n = 12,567)\n(Cell = P(High >= 3), Brackets = P('A great deal'); Entropy = 0.844)", loc="left", pad=10)
    _save(fig, png_dir / "lca_citizen_profiles.png", webp_dir, "lca-citizen-profiles.webp")

    # 9. LCA Citizen Archetype Mix Across 6 G8 Nations
    fig, ax = plt.subplots(figsize=(8.4, 5.9), dpi=220)
    cls_cols = [
        ("Broad Distrustors", "C1: Broad Distrustors", "#c0392b"),
        ("Rule-of-Law Pragmatists", "C2: Rule-of-Law Pragmatists", "#f59e0b"),
        ("Moderate Trustors (Quite a lot)", "C3: Moderate ('Quite a lot')", "#0ea5e9"),
        ("Strong Trustors (A great deal)", "C4: Strong ('A great deal')", "#1e3a8a"),
    ]
    ds = df_lca_s.copy()
    ds["_tot"] = ds["Moderate Trustors (Quite a lot)"] + ds["Strong Trustors (A great deal)"]
    ds = ds.sort_values("_tot", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(ds))
    left = np.zeros(len(ds))
    for col_name, lbl, c in cls_cols:
        vals = ds[col_name].to_numpy()
        ax.barh(y_pos, vals, left=left, height=0.58, color=c, edgecolor="white", linewidth=1.0, label=lbl)
        for i, v in enumerate(vals):
            if v >= 8.0:
                tc = "white" if c in ("#c0392b", "#1e3a8a") else "#0f172a"
                ax.text(left[i] + v / 2.0, y_pos[i], f"{v:.1f}%", ha="center", va="center", fontsize=9.8, fontweight="bold", color=tc)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['country_name']}\n(n={int(r['n_obs']):,})" for _, r in ds.iterrows()], fontsize=10.2, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Posterior Archetype Share (%)")
    ax.set_title("Citizen Archetype Mix Across 6 G8 Nations\n(Posterior Class Shares Across 8 Discrete Trust Items)", loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=2, frameon=False, fontsize=9.4)
    _save(fig, png_dir / "lca_country_archetype_mix.png", webp_dir, "lca-country-archetype-mix.webp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--png-dir", type=Path, default=Path("figures/panels"))
    parser.add_argument("--webp-dir", type=Path, default=None)
    args = parser.parse_args()
    build_all_panels(args.results_dir, args.png_dir, args.webp_dir)
    print("Built all 9 standalone WVS panels cleanly.")
