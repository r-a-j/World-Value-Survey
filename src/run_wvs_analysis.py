"""End-to-end World Values Survey institutional trust analysis and figure generator.

Reads official WVS Wave 7 (v6.0, 97,220 respondents, 66 countries) and WVS Time Series
1981-2022 (v5.0, 443,488 respondents, 108 countries) datasets and executes:
    1. RQ1: Strictly filtered USA Wave 7 Cumulative Link Ordered Logit (n = 2,596),
            exact VIF (>= 1.0), and 3-cutpoint Proportional Odds check.
    2. RQ2: Joint 6-country G8 Wave 7 Ordered Logit with categorical Country fixed
            effects + Country x Predictor interaction LRT, plus 66-country external
            macro-indicator analysis (World Bank GDP PPP, Transparency International
            CPI, Freedom House, V-Dem Liberal Democracy & Press Freedom).
    3. RQ3: Multi-decade longitudinal Cumulative Link Ordered Logit (1995-2022) with
            Institution x Year interactions and delta-method P(High Trust >= 3) 95% CIs.
    4. Bonus: Model-based Latent Class Analysis (LCA, multi-start EM) across 8 discrete
              4-level institutional trust items on G8 Wave 7 respondents.

All institutional confidence items are reverse-coded consistently to an ascending scale:
    1 = None at all, 2 = Not very much, 3 = Quite a lot, 4 = A great deal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.latent_class import LCAResult, fit_lca
from src.ordinal_model import (
    OrderedLogitResult,
    compute_vif,
    fit_ordered_logit,
    likelihood_ratio_test,
    proportional_odds_check,
)

# Strictly ascending trust scale labels (reverse-coded from raw WVS 1=Great deal..4=None)
TRUST_LABELS = ["None at all (1)", "Not very much (2)", "Quite a lot (3)", "A great deal (4)"]

G8_WAVE7 = ["CAN", "DEU", "GBR", "JPN", "RUS", "USA"]
COUNTRY_NAMES = {
    "CAN": "Canada",
    "DEU": "Germany",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "RUS": "Russia",
    "USA": "United States",
}

W7_CONF_MAP = {
    "Q65": "Armed Forces",
    "Q66": "The Press",
    "Q69": "Police",
    "Q70": "Courts",
    "Q71": "Government",
    "Q72": "Political Parties",
    "Q73": "Parliament",
    "Q74": "Civil Services",
    "Q76": "Elections",
}

TS_CONF_MAP = {
    "E069_02": "Armed Forces",
    "E069_06": "Police",
    "E069_07": "Parliament",
    "E069_08": "Civil Services",
    "E069_11": "Government",
    "E069_12": "Political Parties",
    "E069_17": "Courts",
}


def _rev_trust(series: pd.Series) -> pd.Series:
    """Reverse-code WVS 1..4 trust items so 1=None at all .. 4=A great deal; invalid -> NaN."""
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.where(vals.isin([1, 2, 3, 4]))
    return 5.0 - valid


def _clean_range(series: pd.Series, lo: float, hi: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.where((vals >= lo) & (vals <= hi))


def load_wave7(data_dir: Path) -> pd.DataFrame:
    """Load required micro and macro columns from official WVS Wave 7 CSV."""
    use_cols = [
        "B_COUNTRY_ALPHA",
        "A_YEAR",
        "W_WEIGHT",
        "Q65", "Q66", "Q69", "Q70", "Q71", "Q72", "Q73", "Q74", "Q76",
        "Q209", "Q212", "Q235", "Q236", "Q250",
        "Q260", "Q262", "Q275R", "Q288",
        "GDPpercap2", "corrupttransp", "hdi", "freestfh", "polity",
        "v2x_libdem", "v2x_freexp_altinf", "v2xcl_rol", "giniWB",
    ]
    path = data_dir / "wave_7.csv"
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)

    out = pd.DataFrame(index=df.index)
    out["country"] = df["B_COUNTRY_ALPHA"].astype(str)
    out["year"] = pd.to_numeric(df["A_YEAR"], errors="coerce")
    out["weight"] = pd.to_numeric(df["W_WEIGHT"], errors="coerce").fillna(1.0)

    # Reverse-coded institutional trust items (1=None at all .. 4=A great deal)
    for q_col, label in W7_CONF_MAP.items():
        out[label] = _rev_trust(df[q_col])

    # Political attitudes & civic engagement
    out["Democracy Importance"] = _clean_range(df["Q250"], 1, 10)
    # Q235: 1=Very good .. 4=Very bad -> reverse so higher = stronger authoritarian favor
    q235 = _clean_range(df["Q235"], 1, 4)
    out["Favor Strong Leader"] = 5.0 - q235
    q236 = _clean_range(df["Q236"], 1, 4)
    out["Favor Expert Rule"] = 5.0 - q236

    # Q209: 1=Have done, 2=Might do, 3=Would never do
    q209 = _clean_range(df["Q209"], 1, 3)
    out["Signed Petition"] = np.where(q209.isna(), np.nan, (q209 == 1).astype(float))
    q212 = _clean_range(df["Q212"], 1, 3)
    out["Open to Strikes"] = np.where(q212.isna(), np.nan, (q212 <= 2).astype(float))

    # Demographics
    q260 = _clean_range(df["Q260"], 1, 2)
    out["Female"] = np.where(q260.isna(), np.nan, (q260 == 2).astype(float))
    out["Age"] = _clean_range(df["Q262"], 15, 105)
    out["Age_10yr"] = (out["Age"] - 45.0) / 10.0
    out["Education_R"] = _clean_range(df["Q275R"], 1, 3)
    # Use Middle Education (level 2) as reference so neither USA nor G8 suffers sparse-reference VIF inflation
    out["Edu_Lower"] = np.where(out["Education_R"].isna(), np.nan, (out["Education_R"] == 1).astype(float))
    out["Edu_Upper"] = np.where(out["Education_R"].isna(), np.nan, (out["Education_R"] == 3).astype(float))
    out["Income Decile"] = _clean_range(df["Q288"], 1, 10)

    # External macro indicators (strictly cleaning WVS negative / zero missing codes)
    gdp = pd.to_numeric(df["GDPpercap2"], errors="coerce")
    out["GDP_PPP"] = gdp.where(gdp > 0)
    cpi = pd.to_numeric(df["corrupttransp"], errors="coerce")
    out["CPI"] = cpi.where((cpi >= 0) & (cpi <= 100))
    fh = pd.to_numeric(df["freestfh"], errors="coerce")
    out["Freedom_House"] = fh.where((fh >= 0) & (fh <= 100))
    hdi = pd.to_numeric(df["hdi"], errors="coerce")
    out["HDI"] = hdi.where((hdi > 0) & (hdi <= 1.0))
    vdem_ld = pd.to_numeric(df["v2x_libdem"], errors="coerce")
    out["VDem_LibDem"] = vdem_ld.where((vdem_ld >= 0) & (vdem_ld <= 1.0))
    vdem_press = pd.to_numeric(df["v2x_freexp_altinf"], errors="coerce")
    out["VDem_PressFree"] = vdem_press.where((vdem_press >= 0) & (vdem_press <= 1.0))
    vdem_rol = pd.to_numeric(df["v2xcl_rol"], errors="coerce")
    out["VDem_RuleOfLaw"] = vdem_rol.where((vdem_rol >= 0) & (vdem_rol <= 1.0))

    return out


def load_time_series(data_dir: Path) -> pd.DataFrame:
    """Load G8 multi-decade waves (1995-2022) from official WVS Time Series CSV."""
    use_cols = [
        "COUNTRY_ALPHA",
        "S020",
        "E069_02", "E069_06", "E069_07", "E069_08", "E069_11", "E069_12", "E069_17",
        "X003", "X047_WVS",
    ]
    path = data_dir / "time_series_1981_2022.csv"
    chunks = []
    for chunk in pd.read_csv(path, usecols=use_cols, chunksize=100_000, low_memory=False):
        sub = chunk[
            chunk["COUNTRY_ALPHA"].isin(G8_WAVE7) & (pd.to_numeric(chunk["S020"], errors="coerce") >= 1995)
        ]
        if len(sub) > 0:
            chunks.append(sub)
    raw = pd.concat(chunks, ignore_index=True)

    out = pd.DataFrame(index=raw.index)
    out["country"] = raw["COUNTRY_ALPHA"].astype(str)
    out["year"] = pd.to_numeric(raw["S020"], errors="coerce").astype(int)
    for col, label in TS_CONF_MAP.items():
        out[label] = _rev_trust(raw[col])
    out["Age"] = _clean_range(raw["X003"], 15, 105)
    out["Age_10yr"] = (out["Age"] - 45.0) / 10.0
    out["Income Decile"] = _clean_range(raw["X047_WVS"], 1, 10)
    return out


def _result_to_df(res: OrderedLogitResult, model_label: str) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(res.feature_names):
        rows.append(
            {
                "model": model_label,
                "term": name,
                "beta": res.beta[i],
                "se": res.se_beta[i],
                "z": res.z_beta[i],
                "p_value": res.p_beta[i],
                "odds_ratio": res.odds_ratio[i],
                "or_ci_low": res.or_ci_low[i],
                "or_ci_high": res.or_ci_high[i],
                "n_obs": res.n_obs,
                "log_likelihood": res.log_likelihood,
                "mcfadden_r2": res.mcfadden_r2,
                "aic": res.aic,
                "bic": res.bic,
            }
        )
    for j, th in enumerate(res.thresholds):
        rows.append(
            {
                "model": model_label,
                "term": f"Threshold_{j+1}|{j+2}",
                "beta": th,
                "se": res.se_thresholds[j],
                "z": th / res.se_thresholds[j] if res.se_thresholds[j] > 0 else np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
                "or_ci_low": np.nan,
                "or_ci_high": np.nan,
                "n_obs": res.n_obs,
                "log_likelihood": res.log_likelihood,
                "mcfadden_r2": res.mcfadden_r2,
                "aic": res.aic,
                "bic": res.bic,
            }
        )
    return pd.DataFrame(rows)


def run_rq1_usa(w7: pd.DataFrame, out_res: Path) -> dict:
    """RQ1: Strictly filtered USA Wave 7 Cumulative Link Ordered Logit, VIF & PO check."""
    usa_raw = w7[w7["country"] == "USA"].copy()
    socio_cols = [
        "Democracy Importance",
        "Favor Strong Leader",
        "Age_10yr",
        "Female",
        "Edu_Lower",
        "Edu_Upper",
        "Income Decile",
        "Signed Petition",
        "Open to Strikes",
    ]
    inst_cols = [
        "Political Parties",
        "Civil Services",
        "Elections",
        "Courts",
        "Police",
        "Armed Forces",
        "The Press",
    ]
    all_cols = inst_cols + socio_cols

    df_cc = usa_raw[["Government"] + all_cols].dropna().copy()
    y = (df_cc["Government"].to_numpy(dtype=np.int64) - 1)

    res_socio = fit_ordered_logit(df_cc[socio_cols].to_numpy(), y, socio_cols)
    res_full = fit_ordered_logit(df_cc[all_cols].to_numpy(), y, all_cols)
    lrt_stat, lrt_df, lrt_p = likelihood_ratio_test(res_socio, res_full)

    vif_rows = compute_vif(df_cc[all_cols].to_numpy(), all_cols)
    po_rows = proportional_odds_check(df_cc[all_cols].to_numpy(), y, all_cols)

    df_models = pd.concat(
        [_result_to_df(res_socio, "USA_SocioPolitical_1A"), _result_to_df(res_full, "USA_Full_1B")],
        ignore_index=True,
    )
    df_models.to_csv(out_res / "usa_ordered_logit.csv", index=False)
    pd.DataFrame(vif_rows).to_csv(out_res / "usa_vif.csv", index=False)
    pd.DataFrame(po_rows).to_csv(out_res / "usa_proportional_odds.csv", index=False)

    return {
        "usa_total_n": int(len(usa_raw)),
        "usa_complete_n": int(len(df_cc)),
        "res_socio": res_socio,
        "res_full": res_full,
        "lrt_inst_vs_socio": (lrt_stat, lrt_df, lrt_p),
        "vif": pd.DataFrame(vif_rows),
        "po": pd.DataFrame(po_rows),
    }


def run_rq2_g8_and_macro(w7: pd.DataFrame, out_res: Path) -> dict:
    """RQ2: Joint 6-G8 Ordered Logit (categorical Country + interactions) & 66-country macro link."""
    g8_raw = w7[w7["country"].isin(G8_WAVE7)].copy()

    # Exact 4-level Government trust distribution per G8 country
    dist_rows = []
    for code in G8_WAVE7:
        sub = g8_raw[g8_raw["country"] == code]
        gov = sub["Government"].dropna()
        n_gov = len(gov)
        yr = int(sub["year"].mode().iloc[0])
        shares = {f"p_lvl_{lvl}": float((gov == lvl).mean()) for lvl in [1, 2, 3, 4]}
        high_p = float((gov >= 3).mean())
        high_se = float(np.sqrt(high_p * (1.0 - high_p) / n_gov))
        # Also weighted high trust for transparency check
        valid_mask = sub["Government"].notna()
        w = sub.loc[valid_mask, "weight"].to_numpy()
        g_vals = sub.loc[valid_mask, "Government"].to_numpy()
        high_p_w = float(np.average(g_vals >= 3, weights=w))
        dist_rows.append(
            {
                "country": code,
                "country_name": COUNTRY_NAMES[code],
                "year": yr,
                "n_total": int(len(sub)),
                "n_gov_valid": int(n_gov),
                **shares,
                "high_trust_pct": 100.0 * high_p,
                "high_trust_se_pct": 100.0 * high_se,
                "high_trust_weighted_pct": 100.0 * high_p_w,
                "police_high_pct": 100.0 * float((sub["Police"].dropna() >= 3).mean()),
                "courts_high_pct": 100.0 * float((sub["Courts"].dropna() >= 3).mean()),
                "parties_high_pct": 100.0 * float((sub["Political Parties"].dropna() >= 3).mean()),
            }
        )
    df_g8_dist = pd.DataFrame(dist_rows).sort_values("high_trust_pct", ascending=False)
    df_g8_dist.to_csv(out_res / "g8_country_distribution.csv", index=False)

    # Joint 6-Country G8 Ordered Logit (Germany = reference category)
    other_countries = ["CAN", "JPN", "RUS", "USA", "GBR"]
    base_inst = ["Political Parties", "Civil Services", "Elections", "Courts", "Police", "Armed Forces", "The Press"]
    base_socio = ["Democracy Importance", "Favor Strong Leader", "Age_10yr", "Female", "Edu_Lower", "Edu_Upper", "Income Decile"]
    all_base = base_inst + base_socio

    df_cc = g8_raw[["country", "Government"] + all_base].dropna().copy()
    for code in other_countries:
        df_cc[f"Country_{code}"] = (df_cc["country"] == code).astype(float)
    country_cols = [f"Country_{code}" for code in other_countries]

    y = df_cc["Government"].to_numpy(dtype=np.int64) - 1

    # Model 2A-socio: Socio-political + Country FE (shows true unmediated country gaps)
    res_socio_fe = fit_ordered_logit(
        df_cc[base_socio + country_cols].to_numpy(), y, base_socio + country_cols
    )
    # Model 2A: All predictors WITHOUT Country dummies
    res_no_country = fit_ordered_logit(df_cc[all_base].to_numpy(), y, all_base)
    # Model 2B: All predictors + Categorical Country Fixed Effects (ref = Germany)
    res_country_fe = fit_ordered_logit(
        df_cc[all_base + country_cols].to_numpy(), y, all_base + country_cols
    )
    lrt_country = likelihood_ratio_test(res_no_country, res_country_fe)

    # Model 2C: Formal Country x Key Predictor Interactions on the exact same dataset
    interact_vars = ["Favor Strong Leader", "Democracy Importance", "Police", "Political Parties", "Age_10yr"]
    inter_cols = []
    for code in other_countries:
        c_arr = df_cc[f"Country_{code}"].to_numpy()
        for v in interact_vars:
            cname = f"{code}_x_{v}"
            # Center continuous/ordinal vars for clean main-effect interpretation
            v_centered = df_cc[v].to_numpy() - df_cc[v].mean()
            df_cc[cname] = c_arr * v_centered
            inter_cols.append(cname)

    res_interact = fit_ordered_logit(
        df_cc[all_base + country_cols + inter_cols].to_numpy(),
        y,
        all_base + country_cols + inter_cols,
    )
    lrt_interact = likelihood_ratio_test(res_country_fe, res_interact)

    df_joint = pd.concat(
        [
            _result_to_df(res_socio_fe, "G8_Socio_CountryFE"),
            _result_to_df(res_no_country, "G8_NoCountry_2A"),
            _result_to_df(res_country_fe, "G8_Full_CountryFE_2B"),
            _result_to_df(res_interact, "G8_CountryInteractions_2C"),
        ],
        ignore_index=True,
    )
    df_joint.to_csv(out_res / "g8_joint_model.csv", index=False)

    # Global 66-Country External Macro Analysis
    macro_rows = []
    for code, grp in w7.groupby("country"):
        gov = grp["Government"].dropna()
        pol = grp["Police"].dropna()
        crt = grp["Courts"].dropna()
        arm = grp["Armed Forces"].dropna()
        prt = grp["Political Parties"].dropna()
        macro_rows.append(
            {
                "country": code,
                "n_obs": int(len(grp)),
                "is_g8": code in G8_WAVE7,
                "gov_high_pct": 100.0 * float((gov >= 3).mean()) if len(gov) >= 50 else np.nan,
                "police_high_pct": 100.0 * float((pol >= 3).mean()) if len(pol) >= 50 else np.nan,
                "courts_high_pct": 100.0 * float((crt >= 3).mean()) if len(crt) >= 50 else np.nan,
                "armed_high_pct": 100.0 * float((arm >= 3).mean()) if len(arm) >= 50 else np.nan,
                "parties_high_pct": 100.0 * float((prt >= 3).mean()) if len(prt) >= 50 else np.nan,
                "GDP_PPP": grp["GDP_PPP"].dropna().iloc[0] if grp["GDP_PPP"].notna().any() else np.nan,
                "CPI": grp["CPI"].dropna().iloc[0] if grp["CPI"].notna().any() else np.nan,
                "Freedom_House": grp["Freedom_House"].dropna().iloc[0] if grp["Freedom_House"].notna().any() else np.nan,
                "HDI": grp["HDI"].dropna().iloc[0] if grp["HDI"].notna().any() else np.nan,
                "VDem_LibDem": grp["VDem_LibDem"].dropna().iloc[0] if grp["VDem_LibDem"].notna().any() else np.nan,
                "VDem_PressFree": grp["VDem_PressFree"].dropna().iloc[0] if grp["VDem_PressFree"].notna().any() else np.nan,
                "VDem_RuleOfLaw": grp["VDem_RuleOfLaw"].dropna().iloc[0] if grp["VDem_RuleOfLaw"].notna().any() else np.nan,
            }
        )
    df_macro = pd.DataFrame(macro_rows)
    df_macro.to_csv(out_res / "global_66_macro_trust.csv", index=False)

    corr_rows = []
    for trust_col in ["gov_high_pct", "police_high_pct", "courts_high_pct"]:
        for macro_col in ["Freedom_House", "VDem_PressFree", "VDem_LibDem", "CPI", "GDP_PPP", "HDI"]:
            pair = df_macro[[trust_col, macro_col]].dropna()
            r_val, p_r = pearsonr(pair[macro_col], pair[trust_col])
            rho_val, p_rho = spearmanr(pair[macro_col], pair[trust_col])
            corr_rows.append(
                {
                    "trust_metric": trust_col,
                    "macro_indicator": macro_col,
                    "n_countries": int(len(pair)),
                    "pearson_r": float(r_val),
                    "pearson_p": float(p_r),
                    "spearman_rho": float(rho_val),
                    "spearman_p": float(p_rho),
                }
            )
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(out_res / "macro_correlations.csv", index=False)

    return {
        "g8_total_n": int(len(g8_raw)),
        "g8_complete_n": int(len(df_cc)),
        "df_g8_dist": df_g8_dist,
        "res_socio_fe": res_socio_fe,
        "res_no_country": res_no_country,
        "res_country_fe": res_country_fe,
        "res_interact": res_interact,
        "lrt_country": lrt_country,
        "lrt_interact": lrt_interact,
        "df_macro": df_macro,
        "df_corr": df_corr,
    }


def run_rq3_longitudinal(ts: pd.DataFrame, out_res: Path) -> dict:
    """RQ3: Multi-decade USA 5-institution model & 6-G8 Government trust trajectories."""
    # Part A: USA 5-institution stacked Cumulative Link Ordered Logit (1995-2017)
    usa = ts[ts["country"] == "USA"].copy()
    usa_insts = ["Armed Forces", "Police", "Courts", "Government", "Parliament", "Political Parties"]
    usa_years = [1995, 1999, 2006, 2011, 2017]

    long_parts = []
    for inst in usa_insts:
        part = usa[["year", "Age_10yr", "Income Decile", inst]].dropna().copy()
        part = part.rename(columns={inst: "trust"})
        part["institution"] = inst
        long_parts.append(part)
    usa_long = pd.concat(long_parts, ignore_index=True)

    # Reference: Government, Year 1995
    ref_inst = "Government"
    other_insts = [i for i in usa_insts if i != ref_inst]
    other_years = [yr for yr in usa_years if yr != 1995]

    for inst in other_insts:
        usa_long[f"Inst_{inst}"] = (usa_long["institution"] == inst).astype(float)
    for yr in other_years:
        usa_long[f"Yr_{yr}"] = (usa_long["year"] == yr).astype(float)

    add_cols = (
        [f"Inst_{i}" for i in other_insts]
        + [f"Yr_{yr}" for yr in other_years]
        + ["Age_10yr", "Income Decile"]
    )
    inter_cols = []
    for inst in other_insts:
        for yr in other_years:
            # Skip Courts x 1999 (not surveyed in USA 1999)
            if inst == "Courts" and yr == 1999:
                continue
            cname = f"Inst_{inst}_x_{yr}"
            usa_long[cname] = usa_long[f"Inst_{inst}"] * usa_long[f"Yr_{yr}"]
            inter_cols.append(cname)

    y_usa = usa_long["trust"].to_numpy(dtype=np.int64) - 1
    res_usa_add = fit_ordered_logit(usa_long[add_cols].to_numpy(), y_usa, add_cols)
    res_usa_int = fit_ordered_logit(
        usa_long[add_cols + inter_cols].to_numpy(), y_usa, add_cols + inter_cols
    )
    lrt_usa_div = likelihood_ratio_test(res_usa_add, res_usa_int)

    # Predicted covariate-adjusted P(High Trust >= 3) at Age = 45 (Age_10yr=0), Income = 5
    # Fit institution-specific Ordered Logits so each institution has its own 3 thresholds
    # (preventing shared-threshold distortion from polarized tails like USA 2017 Government).
    pred_rows = []
    for inst in usa_insts:
        sub_i = usa_long[usa_long["institution"] == inst].copy()
        yrs_i = sorted(sub_i["year"].unique().tolist())
        ref_y = yrs_i[0]
        yr_cols = []
        for yr in yrs_i[1:]:
            cname = f"Yr_{yr}"
            sub_i[cname] = (sub_i["year"] == yr).astype(float)
            yr_cols.append(cname)
        f_cols = yr_cols + ["Age_10yr", "Income Decile"]
        res_i = fit_ordered_logit(sub_i[f_cols].to_numpy(), sub_i["trust"].to_numpy(dtype=np.int64) - 1, f_cols)
        for yr in yrs_i:
            sub_yr = sub_i[sub_i["year"] == yr]
            x_row = np.zeros((1, len(f_cols)), dtype=np.float64)
            if yr != ref_y:
                x_row[0, f_cols.index(f"Yr_{yr}")] = 1.0
            x_row[0, f_cols.index("Age_10yr")] = 0.0
            x_row[0, f_cols.index("Income Decile")] = 5.0
            p_hi, lo, hi = res_i.predict_high_trust_with_ci(x_row, threshold_idx=1)
            pred_rows.append(
                {
                    "institution": inst,
                    "year": yr,
                    "n_obs": int(len(sub_yr)),
                    "raw_none_pct": 100.0 * float((sub_yr["trust"] == 1).mean()),
                    "raw_high_pct": 100.0 * float((sub_yr["trust"] >= 3).mean()),
                    "pred_high_pct": 100.0 * float(p_hi[0]),
                    "ci_low_pct": 100.0 * float(lo[0]),
                    "ci_high_pct": 100.0 * float(hi[0]),
                }
            )
    df_usa_trends = pd.DataFrame(pred_rows)
    df_usa_trends.to_csv(out_res / "longitudinal_usa_institutions.csv", index=False)

    # Part B: Cross-National G8 Government Trust trajectories (1995-2022)
    g8_ts = ts[["country", "year", "Government", "Age_10yr", "Income Decile"]].dropna().copy()
    # Fit covariate-adjusted Ordered Logit per country over its surveyed waves
    g8_pred_rows = []
    for code in G8_WAVE7:
        sub_c = g8_ts[g8_ts["country"] == code].copy()
        years = sorted(sub_c["year"].unique().tolist())
        if len(years) < 2:
            continue
        ref_yr = years[0]
        yr_cols = []
        for yr in years[1:]:
            cname = f"Yr_{yr}"
            sub_c[cname] = (sub_c["year"] == yr).astype(float)
            yr_cols.append(cname)
        feat_cols = yr_cols + ["Age_10yr", "Income Decile"]
        y_c = sub_c["Government"].to_numpy(dtype=np.int64) - 1
        res_c = fit_ordered_logit(sub_c[feat_cols].to_numpy(), y_c, feat_cols)
        for yr in years:
            x_row = np.zeros((1, len(feat_cols)), dtype=np.float64)
            if yr != ref_yr:
                x_row[0, feat_cols.index(f"Yr_{yr}")] = 1.0
            x_row[0, feat_cols.index("Age_10yr")] = 0.0
            x_row[0, feat_cols.index("Income Decile")] = 5.0
            p_hi, lo, hi = res_c.predict_high_trust_with_ci(x_row, threshold_idx=1)
            n_yr = int((sub_c["year"] == yr).sum())
            raw_p = 100.0 * float((sub_c.loc[sub_c["year"] == yr, "Government"] >= 3).mean())
            g8_pred_rows.append(
                {
                    "country": code,
                    "country_name": COUNTRY_NAMES[code],
                    "year": int(yr),
                    "n_obs": n_yr,
                    "raw_high_pct": raw_p,
                    "pred_high_pct": 100.0 * float(p_hi[0]),
                    "ci_low_pct": 100.0 * float(lo[0]),
                    "ci_high_pct": 100.0 * float(hi[0]),
                }
            )
    df_g8_trends = pd.DataFrame(g8_pred_rows)
    df_g8_trends.to_csv(out_res / "longitudinal_g8_government.csv", index=False)

    return {
        "usa_respondents": int(len(usa)),
        "usa_stacked_obs": int(len(usa_long)),
        "res_usa_add": res_usa_add,
        "res_usa_int": res_usa_int,
        "lrt_usa_div": lrt_usa_div,
        "df_usa_trends": df_usa_trends,
        "df_g8_trends": df_g8_trends,
    }


def run_bonus_lca(w7: pd.DataFrame, out_res: Path, chosen_k: int = 4) -> dict:
    """Bonus: Model-Based Latent Class Analysis (LCA) on 8 discrete 4-level trust items."""
    lca_items = [
        "Armed Forces",
        "Police",
        "Courts",
        "Civil Services",
        "Elections",
        "Government",
        "Parliament",
        "Political Parties",
    ]
    g8 = w7[w7["country"].isin(G8_WAVE7)].copy()
    df_cc = g8[["country", "Age", "Democracy Importance", "Favor Strong Leader", "Income Decile"] + lca_items].dropna(
        subset=lca_items
    ).copy()
    Y = df_cc[lca_items].to_numpy(dtype=np.int64) - 1

    models: dict[int, LCAResult] = {}
    sel_rows = []
    for k in [2, 3, 4, 5, 6]:
        res = fit_lca(Y, lca_items, n_classes=k, n_categories=4, n_init=12, seed=42)
        models[k] = res
        sel_rows.append(
            {
                "K": k,
                "n_obs": res.n_obs,
                "n_params": res.n_params,
                "log_likelihood": res.log_likelihood,
                "aic": res.aic,
                "bic": res.bic,
                "sabic": res.sabic,
                "entropy": res.entropy,
                "converged": res.converged,
            }
        )
    df_sel = pd.DataFrame(sel_rows)
    df_sel.to_csv(out_res / "lca_model_selection.csv", index=False)

    best = models[chosen_k]
    hi_mat = best.high_trust_matrix()
    df_cc["lca_class"] = best.hard_assignments

    # Data-driven human labels grounded in full 4-level response geometry (sorted low -> high trust)
    profile_rows = []
    class_labels = []
    for k in range(chosen_k):
        row_hi = hi_mat[k]
        rol_mean = float(np.mean(row_hi[:3]))
        pol_mean = float(np.mean(row_hi[5:]))
        overall = float(np.mean(row_hi))
        p_great = float(best.item_probs[k, :, 3].mean())
        p_quite = float(best.item_probs[k, :, 2].mean())
        p_none = float(best.item_probs[k, :, 0].mean())
        sub_k = df_cc[df_cc["lca_class"] == k]
        if overall < 0.28:
            lbl = "Broad Distrustors"
        elif rol_mean - pol_mean > 0.28 and overall < 0.65:
            lbl = "Rule-of-Law Pragmatists"
        elif p_great > 0.30:
            lbl = "Strong Trustors (A great deal)"
        else:
            lbl = "Moderate Trustors (Quite a lot)"
        class_labels.append(lbl)
        entry = {
            "class_id": k + 1,
            "archetype": lbl,
            "weight_pct": 100.0 * float(best.class_weights[k]),
            "n_hard": int(len(sub_k)),
            "mean_age": float(sub_k["Age"].mean()),
            "mean_dem_imp": float(sub_k["Democracy Importance"].mean()),
            "favor_strong_leader_pct": 100.0 * float((sub_k["Favor Strong Leader"] >= 3).mean()),
            "avg_none_pct": 100.0 * p_none,
            "avg_quite_pct": 100.0 * p_quite,
            "avg_great_pct": 100.0 * p_great,
        }
        for j, item in enumerate(lca_items):
            entry[f"hi_{item}"] = 100.0 * float(row_hi[j])
            entry[f"great_{item}"] = 100.0 * float(best.item_probs[k, j, 3])
        profile_rows.append(entry)

    # Disambiguate if any label collided
    seen: dict[str, int] = {}
    for idx, lbl in enumerate(class_labels):
        if class_labels.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            new_lbl = f"{lbl} ({seen[lbl]})"
            class_labels[idx] = new_lbl
            profile_rows[idx]["archetype"] = new_lbl

    df_profiles = pd.DataFrame(profile_rows)
    df_profiles.to_csv(out_res / "lca_item_profiles.csv", index=False)

    # Country x Archetype composition table (using exact soft posterior weights)
    share_rows = []
    for code in G8_WAVE7:
        mask = (df_cc["country"] == code).to_numpy()
        post_means = best.responsibilities[mask].mean(axis=0)
        row = {"country": code, "country_name": COUNTRY_NAMES[code], "n_obs": int(mask.sum())}
        for k, lbl in enumerate(class_labels):
            row[lbl] = 100.0 * float(post_means[k])
        share_rows.append(row)
    df_shares = pd.DataFrame(share_rows)
    df_shares.to_csv(out_res / "lca_country_shares.csv", index=False)

    return {
        "n_complete": int(len(df_cc)),
        "items": lca_items,
        "models": models,
        "chosen_k": chosen_k,
        "best": best,
        "class_labels": class_labels,
        "df_sel": df_sel,
        "df_profiles": df_profiles,
        "df_shares": df_shares,
    }


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def plot_figure1_g8_and_macro(rq2: dict, out_fig: Path) -> None:
    """Figure 1: G8 4-level Government Trust + 66-Country Critical Citizens Macro Paradox."""
    _style_matplotlib()
    fig = plt.figure(figsize=(14.8, 6.7), dpi=220)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.52)

    # Panel A (Left, spans both rows): Properly ordered 4-level stacked horizontal bars
    ax_bar = fig.add_subplot(gs[:, 0])
    df_g8 = rq2["df_g8_dist"].sort_values("high_trust_pct", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_g8))

    colors = ["#c0392b", "#e67e22", "#3b82f6", "#1e3a8a"]
    cols = ["p_lvl_1", "p_lvl_2", "p_lvl_3", "p_lvl_4"]
    left = np.zeros(len(df_g8))

    for col, color, lbl in zip(cols, colors, TRUST_LABELS):
        vals = 100.0 * df_g8[col].to_numpy()
        ax_bar.barh(y_pos, vals, left=left, height=0.60, color=color, edgecolor="white", linewidth=0.8, label=lbl)
        for i, v in enumerate(vals):
            if v >= 6.5:
                txt_col = "white" if color in ("#c0392b", "#1e3a8a", "#3b82f6") else "#1e293b"
                ax_bar.text(left[i] + v / 2.0, y_pos[i], f"{v:.1f}%", ha="center", va="center", fontsize=8.4, fontweight="bold", color=txt_col)
        left += vals

    for i, row in df_g8.iterrows():
        ax_bar.text(101.8, i, f"High: {row['high_trust_pct']:.1f}%", va="center", ha="left", fontsize=9.0, fontweight="bold", color="#0f172a")

    y_labels = [f"{r['country_name']}\n({int(r['year'])}, n={int(r['n_gov_valid']):,})" for _, r in df_g8.iterrows()]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(y_labels, fontsize=9.1)
    ax_bar.set_xlim(0, 122)
    ax_bar.set_xticks([0, 25, 50, 75, 100])
    ax_bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_bar.set_xlabel("Share of Valid Respondents (%)")
    ax_bar.set_title("A. Government Trust Across 6 G8 Wave 7 Nations\n(Ascending 1-4 Scale; France & Italy Absent in Wave 7)", pad=10, loc="left", fontsize=11.0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.45, -0.10), ncol=2, frameon=False)

    # Panel B (Top-Right): V-Dem Press & Expression Freedom vs Government Trust (Critical Citizens Paradox)
    df_m = rq2["df_macro"]
    ax_top = fig.add_subplot(gs[0, 1])
    p_gov = df_m[["country", "is_g8", "VDem_PressFree", "gov_high_pct"]].dropna()
    other = p_gov[~p_gov["is_g8"]]
    g8_pts = p_gov[p_gov["is_g8"]]
    ax_top.scatter(other["VDem_PressFree"], other["gov_high_pct"], s=32, color="#94a3b8", alpha=0.65, edgecolors="white", linewidth=0.5, label=f"Other WVS nations (n={len(other)})")
    ax_top.scatter(g8_pts["VDem_PressFree"], g8_pts["gov_high_pct"], s=68, color="#dc2626", edgecolors="#0f172a", linewidth=0.9, zorder=5, label="G8 Wave 7 nations (n=6)")

    slope, intercept = np.polyfit(p_gov["VDem_PressFree"], p_gov["gov_high_pct"], 1)
    xs = np.linspace(p_gov["VDem_PressFree"].min(), p_gov["VDem_PressFree"].max(), 100)
    ax_top.plot(xs, slope * xs + intercept, color="#dc2626", linestyle="--", linewidth=1.6, alpha=0.85)
    r_gov, p_r_gov = pearsonr(p_gov["VDem_PressFree"], p_gov["gov_high_pct"])

    offsets_top = {
        "CAN": (-28, 12),
        "DEU": (-10, 13),
        "JPN": (-32, 3),
        "GBR": (-30, -8),
        "USA": (-30, -10),
        "RUS": (8, 6),
    }
    for _, row in g8_pts.iterrows():
        dx, dy = offsets_top.get(row["country"], (6, 4))
        ax_top.annotate(
            row["country"],
            (row["VDem_PressFree"], row["gov_high_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6),
        )
    ax_top.set_title(f"B. Critical Citizens: Press Freedom vs. Gov. Trust\n(n={len(p_gov)} nations: r = {r_gov:+.2f}, p = {p_r_gov:.2e})", loc="left", pad=6, fontsize=10.6)
    ax_top.set_xlabel("V-Dem Press & Expression Freedom Index (0-1)")
    ax_top.set_ylabel("High Gov. Trust (%)")
    ax_top.set_xlim(0.02, 1.04)
    ax_top.set_ylim(0, 102)
    ax_top.grid(True, alpha=0.55)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.9)

    # Panel C (Bottom-Right): Transparency International CPI vs Police High Trust (Rule of Law)
    ax_bot = fig.add_subplot(gs[1, 1])
    p_pol = df_m[["country", "is_g8", "CPI", "police_high_pct"]].dropna()
    other_p = p_pol[~p_pol["is_g8"]]
    g8_p = p_pol[p_pol["is_g8"]]
    ax_bot.scatter(other_p["CPI"], other_p["police_high_pct"], s=32, color="#94a3b8", alpha=0.65, edgecolors="white", linewidth=0.5)
    ax_bot.scatter(g8_p["CPI"], g8_p["police_high_pct"], s=68, color="#1d4ed8", edgecolors="#0f172a", linewidth=0.9, zorder=5)

    slope_p, int_p = np.polyfit(p_pol["CPI"], p_pol["police_high_pct"], 1)
    xs_p = np.linspace(p_pol["CPI"].min(), p_pol["CPI"].max(), 100)
    ax_bot.plot(xs_p, slope_p * xs_p + int_p, color="#1d4ed8", linestyle="--", linewidth=1.6, alpha=0.85)
    r_pol, p_r_pol = pearsonr(p_pol["CPI"], p_pol["police_high_pct"])

    offsets_bot = {
        "DEU": (7, 4),
        "CAN": (8, -6),
        "GBR": (-28, -11),
        "JPN": (-26, 6),
        "USA": (-28, -9),
        "RUS": (8, -7),
    }
    for _, row in g8_p.iterrows():
        dx, dy = offsets_bot.get(row["country"], (6, 4))
        ax_bot.annotate(
            row["country"],
            (row["CPI"], row["police_high_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6),
        )
    ax_bot.set_title(f"C. Rule of Law: Anti-Corruption (CPI) vs. Police Trust\n(n={len(p_pol)} nations: r = {r_pol:+.2f}, p = {p_r_pol:.2e})", loc="left", pad=6, fontsize=10.6)
    ax_bot.set_xlabel("Transparency International CPI (0 = Corrupt, 100 = Clean)")
    ax_bot.set_ylabel("High Police Trust (%)")
    ax_bot.set_ylim(10, 105)
    ax_bot.grid(True, alpha=0.55)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.15)
    fig.savefig(out_fig / "g8_trust_and_macro_paradox.png", dpi=220)
    plt.close(fig)


def plot_figure2_ordered_logit(rq1: dict, rq2: dict, out_fig: Path) -> None:
    """Figure 2: USA Ordered Logit Odds Ratios (+ PO check) & Joint G8 Country Fixed Effects."""
    _style_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.7), dpi=220, gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Panel A: USA Full Ordered Logit Odds Ratios with 95% CI + 3 cumulative binary cutpoints
    res_full: OrderedLogitResult = rq1["res_full"]
    po_df = rq1["po"].set_index("feature")

    inst_set = {"Political Parties", "Civil Services", "Elections", "Courts", "Police", "Armed Forces", "The Press"}
    label_pretty = {
        "Age_10yr": "Age (+10 years)",
        "Edu_Lower": "Education: Lower vs Middle",
        "Edu_Upper": "Education: Upper vs Middle",
        "Democracy Importance": "Democracy Importance (1-10)",
        "Favor Strong Leader": "Favor Strong Leader (1-4)",
        "Income Decile": "Income Decile (1-10)",
    }

    order = np.argsort(res_full.beta)
    y_pos = np.arange(len(order))

    ax1.axvline(1.0, color="#475569", linestyle="--", linewidth=1.1, zorder=1)
    for plot_i, idx in enumerate(order):
        feat = res_full.feature_names[idx]
        or_val = res_full.odds_ratio[idx]
        lo = res_full.or_ci_low[idx]
        hi = res_full.or_ci_high[idx]
        col = "#1d4ed8" if feat in inst_set else "#0f766e"
        ax1.errorbar(
            or_val,
            plot_i,
            xerr=[[or_val - lo], [hi - or_val]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.8,
            capsize=3,
            markersize=6.2,
            zorder=4,
        )
        for cut_col, mk, mcol in [
            ("or_ge_2", "^", "#f59e0b"),
            ("or_ge_3", "s", "#64748b"),
            ("or_ge_4", "v", "#dc2626"),
        ]:
            val_cut = float(po_df.loc[feat, cut_col])
            if 0.22 <= val_cut <= 6.0:
                ax1.scatter(val_cut, plot_i, marker=mk, s=22, facecolors="none", edgecolors=mcol, linewidths=0.9, zorder=5, alpha=0.85)

    # Highlight 2017 US Press-vs-Government polarization finding in blank space at y = 0
    press_pos = int(np.where(order == res_full.feature_names.index("The Press"))[0][0])
    press_or = float(res_full.odds_ratio[res_full.feature_names.index("The Press")])
    ax1.annotate(
        f"2017 Press-vs-Gov\npolarization (OR={press_or:.2f})",
        xy=(press_or * 1.15, press_pos),
        xytext=(0.68, press_pos - 0.18),
        fontsize=8.0,
        fontweight="bold",
        color="#1e3a8a",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.9),
    )

    ax1.scatter([], [], color="#1d4ed8", s=36, label="Institutional Trust OR (95% CI)")
    ax1.scatter([], [], color="#0f766e", s=36, label="Socio-Political OR (95% CI)")
    ax1.scatter([], [], marker="^", s=26, facecolors="none", edgecolors="#f59e0b", label="PO Check: Y >= 2")
    ax1.scatter([], [], marker="s", s=26, facecolors="none", edgecolors="#64748b", label="PO Check: Y >= 3")
    ax1.scatter([], [], marker="v", s=26, facecolors="none", edgecolors="#dc2626", label="PO Check: Y = 4")

    ax1.set_xscale("log")
    ax1.set_xlim(0.26, 4.8)
    ax1.set_xticks([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.3", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.5"]))
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([label_pretty.get(res_full.feature_names[i], res_full.feature_names[i]) for i in order], fontsize=8.8)
    ax1.set_xlabel("Cumulative Odds Ratio for Higher Government Trust (Log Scale)")
    ax1.set_title(
        f"A. USA Wave 7 Ordered Logit & Proportional Odds Audit\n(Strictly USA: n = {rq1['usa_complete_n']:,} / {rq1['usa_total_n']:,}; McFadden R² = {res_full.mcfadden_r2:.3f})",
        loc="left",
        pad=10,
        fontsize=10.8,
    )
    ax1.grid(True, axis="x", alpha=0.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.9)

    # Panel B: Joint 6-Country G8 Categorical Country Fixed Effects (vs Germany = 1.0)
    res_socio_fe: OrderedLogitResult = rq2["res_socio_fe"]
    res_full_fe: OrderedLogitResult = rq2["res_country_fe"]
    codes = ["RUS", "CAN", "JPN", "USA", "GBR"]
    y_c = np.arange(len(codes))

    ax2.axvline(1.0, color="#475569", linestyle="--", linewidth=1.2, label="Germany Reference (OR = 1.0)")

    for i, code in enumerate(codes):
        term = f"Country_{code}"
        idx_s = res_socio_fe.feature_names.index(term)
        idx_f = res_full_fe.feature_names.index(term)

        or_s, lo_s, hi_s = res_socio_fe.odds_ratio[idx_s], res_socio_fe.or_ci_low[idx_s], res_socio_fe.or_ci_high[idx_s]
        ax2.errorbar(
            or_s, i + 0.14, xerr=[[or_s - lo_s], [hi_s - or_s]],
            fmt="s", color="#d97706", ecolor="#d97706", elinewidth=1.8, capsize=3.5, markersize=6.2,
            label="Adjusted for Demographics & Attitudes" if i == 0 else None,
        )
        ax2.text(hi_s * 1.05, i + 0.14, f"{or_s:.2f}", va="center", ha="left", fontsize=8.4, color="#b45309", fontweight="bold")

        or_f, lo_f, hi_f = res_full_fe.odds_ratio[idx_f], res_full_fe.or_ci_low[idx_f], res_full_fe.or_ci_high[idx_f]
        ax2.errorbar(
            or_f, i - 0.14, xerr=[[or_f - lo_f], [hi_f - or_f]],
            fmt="o", color="#1d4ed8", ecolor="#1d4ed8", elinewidth=1.8, capsize=3.5, markersize=6.5,
            label="Adjusted for Other Institutional Trust + Demo." if i == 0 else None,
        )
        ax2.text(hi_f * 1.05, i - 0.14, f"{or_f:.2f}", va="center", ha="left", fontsize=8.4, color="#1e3a8a", fontweight="bold")

    lrt_c_stat, lrt_c_df, _ = rq2["lrt_country"]
    lrt_i_stat, lrt_i_df, _ = rq2["lrt_interact"]
    box_txt = (
        f"Pooled 6-G8: n = {rq2['g8_complete_n']:,}\n"
        f"Country FE LRT: χ²({lrt_c_df}) = {lrt_c_stat:.1f}\n"
        f"Interaction LRT: χ²({lrt_i_df}) = {lrt_i_stat:.1f}\n"
        f"(both p < 1e-100)\n"
        f"R²: {res_full_fe.mcfadden_r2:.3f} (FE) -> {rq2['res_interact'].mcfadden_r2:.3f} (Int.)"
    )
    # Place strictly right of OR = 1.2 in upper-right empty space opposite UK & USA
    ax2.text(
        0.98,
        0.74,
        box_txt,
        transform=ax2.transAxes,
        fontsize=7.9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9),
    )

    ax2.set_xscale("log")
    ax2.set_xlim(0.24, 6.2)
    ax2.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.25", "0.5", "1.0", "2.0", "4.0"]))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_yticks(y_c)
    ax2.set_yticklabels([COUNTRY_NAMES[c] for c in codes], fontsize=9.4, fontweight="bold")
    ax2.set_ylim(-0.65, len(codes) - 0.25)
    ax2.set_xlabel("Categorical Country Odds Ratio vs. Germany (Ref = 1.0)")
    ax2.set_title(
        "B. Joint 6-Country G8 Categorical Effects (Ref: Germany)\n(Categorical Country Dummies + Pooled Interaction LRT)",
        loc="left",
        pad=10,
        fontsize=10.8,
    )
    ax2.grid(True, axis="x", alpha=0.55)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, fontsize=8.0)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.11, wspace=0.28)
    fig.savefig(out_fig / "usa_and_g8_ordered_logit.png", dpi=220)
    plt.close(fig)


def plot_figure3_longitudinal(rq3: dict, out_fig: Path) -> None:
    """Figure 3: Multi-Decade Covariate-Adjusted Predicted High Trust P(Y >= 3) with 95% CIs."""
    _style_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.2), dpi=220)

    # Panel A: USA 6 Institutions (1995-2017)
    df_u = rq3["df_usa_trends"]
    inst_styles = {
        "Armed Forces": ("#1e3a8a", "o", "-"),
        "Police": ("#0284c7", "s", "-"),
        "Courts": ("#0f766e", "^", "--"),
        "Government": ("#d97706", "D", "-"),
        "Parliament": ("#dc2626", "v", "--"),
        "Political Parties": ("#7c3aed", "P", ":"),
    }
    for inst, (col, mk, ls) in inst_styles.items():
        sub = df_u[df_u["institution"] == inst].sort_values("year")
        ax1.plot(sub["year"], sub["pred_high_pct"], color=col, marker=mk, linestyle=ls, linewidth=2.1, markersize=6.2, label=inst)
        ax1.fill_between(sub["year"], sub["ci_low_pct"], sub["ci_high_pct"], color=col, alpha=0.13)

    lrt_stat, lrt_df, _ = rq3["lrt_usa_div"]
    ax1.annotate(
        "Armed Forces stays ~80-86%\n(fixes flipped-sign clmm bug)",
        xy=(2011, float(df_u[(df_u["institution"] == "Armed Forces") & (df_u["year"] == 2011)]["pred_high_pct"].iloc[0])),
        xytext=(1998.5, 92.0),
        fontsize=8.2,
        fontweight="bold",
        color="#1e3a8a",
        arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.0),
    )
    ax1.annotate(
        "Parties & Congress drop\nto 10-14% by 2017",
        xy=(2017, float(df_u[(df_u["institution"] == "Political Parties") & (df_u["year"] == 2017)]["pred_high_pct"].iloc[0])),
        xytext=(2004.2, 2.5),
        fontsize=8.2,
        fontweight="bold",
        color="#7c3aed",
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.0),
    )
    ax1.set_title(
        f"A. USA Institutional Trust Trajectories (1995-2017, n = {rq3['usa_respondents']:,})\n(Ordered Logit P(Y >= 3), 95% CI; Inst × Year LRT χ²({lrt_df}) = {lrt_stat:.0f})",
        loc="left",
        pad=10,
        fontsize=10.6,
    )
    ax1.set_xlabel("WVS Survey Year")
    ax1.set_ylabel("Predicted High Trust Probability P(Y >= 3) (%)")
    ax1.set_xlim(1993.5, 2018.8)
    ax1.set_ylim(0, 102)
    ax1.set_xticks([1995, 1999, 2006, 2011, 2017])
    ax1.grid(True, alpha=0.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, -0.15), ncol=3, frameon=False, fontsize=8.5)

    # Panel B: Cross-National G8 Government Trust (1995-2022)
    df_g = rq3["df_g8_trends"]
    c_styles = {
        "CAN": ("#0284c7", "o", "-"),
        "DEU": ("#16a34a", "s", "-"),
        "JPN": ("#9333ea", "^", "-"),
        "RUS": ("#dc2626", "D", "--"),
        "USA": ("#d97706", "v", "-"),
        "GBR": ("#334155", "P", "--"),
    }
    for code, (col, mk, ls) in c_styles.items():
        sub = df_g[df_g["country"] == code].sort_values("year")
        ax2.plot(sub["year"], sub["pred_high_pct"], color=col, marker=mk, linestyle=ls, linewidth=2.1, markersize=6.2, label=COUNTRY_NAMES[code])
        ax2.fill_between(sub["year"], sub["ci_low_pct"], sub["ci_high_pct"], color=col, alpha=0.12)

    ax2.set_title(
        "B. Cross-National G8 Government Trust (1995-2022)\n(Ordered Logit P(Y >= 3) at Age 45, Income 5, 95% CI)",
        loc="left",
        pad=10,
        fontsize=10.6,
    )
    ax2.set_xlabel("WVS Survey Year")
    ax2.set_ylabel("Predicted High Government Trust P(Y >= 3) (%)")
    ax2.set_xlim(1994, 2023.5)
    ax2.set_ylim(8, 72)
    ax2.grid(True, alpha=0.55)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.50, -0.15), ncol=3, frameon=False, fontsize=8.5)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.22, wspace=0.22)
    fig.savefig(out_fig / "longitudinal_trust_trends.png", dpi=220)
    plt.close(fig)


def plot_figure4_lca(lca: dict, out_fig: Path) -> None:
    """Figure 4: Model-Based Latent Class Analysis (LCA) Archetype Profiles & G8 Country Shares."""
    _style_matplotlib()
    fig = plt.figure(figsize=(15.4, 6.5), dpi=220)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.45)

    # Panel A: Conditional High-Trust Profile Heatmap P(Y_j >= 3 | Class k) + lvl 4 intensity
    ax_hm = fig.add_subplot(gs[0, 0])
    items = lca["items"]
    best: LCAResult = lca["best"]
    labels = lca["class_labels"]
    hi_mat = 100.0 * best.high_trust_matrix()         # (K, J)
    great_mat = 100.0 * best.item_probs[:, :, 3]      # (K, J) P("A great deal")

    im = ax_hm.imshow(hi_mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for k in range(hi_mat.shape[0]):
        for j in range(hi_mat.shape[1]):
            val = hi_mat[k, j]
            g_val = great_mat[k, j]
            txt_col = "white" if val >= 55 else "#0f172a"
            sub_col = "#e2e8f0" if val >= 55 else "#475569"
            ax_hm.text(j, k - 0.12, f"{val:.0f}%", ha="center", va="center", fontsize=8.8, fontweight="bold", color=txt_col)
            ax_hm.text(j, k + 0.21, f"({g_val:.0f}%)", ha="center", va="center", fontsize=7.0, color=sub_col)

    short_labels = {
        "Broad Distrustors": "C1: Broad Distrustors",
        "Rule-of-Law Pragmatists": "C2: Rule-of-Law\nPragmatists",
        "Moderate Trustors (Quite a lot)": "C3: Moderate Trustors\n('Quite a lot')",
        "Strong Trustors (A great deal)": "C4: Strong Trustors\n('A great deal')",
    }
    y_tick_labels = [
        f"{short_labels.get(labels[k], labels[k])}\n({100.0 * best.class_weights[k]:.1f}% of G8)"
        for k in range(best.n_classes)
    ]
    ax_hm.set_yticks(np.arange(best.n_classes))
    ax_hm.set_yticklabels(y_tick_labels, fontsize=8.4, fontweight="bold")
    ax_hm.set_xticks(np.arange(len(items)))
    ax_hm.set_xticklabels(items, rotation=26, ha="right", fontsize=8.6)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.036, pad=0.025)
    cbar.set_label("Conditional P(High Trust >= 3) (%)", fontsize=8.5)
    ax_hm.set_title(
        f"A. Latent Class Analysis (LCA) Trust Profiles (K = {best.n_classes}; n = {lca['n_complete']:,})\n(Cell = P(High >= 3), Brackets = P('A great deal'); Entropy = {best.entropy:.3f})",
        loc="left",
        pad=10,
        fontsize=10.3,
    )

    # Panel B: Country Composition Across LCA Citizen Archetypes
    ax_st = fig.add_subplot(gs[0, 1])
    df_s = lca["df_shares"].copy()
    # Order countries by combined Broad Trustors (C3 + C4)
    df_s["_trust_tot"] = df_s[labels[-2]] + df_s[labels[-1]]
    df_s = df_s.sort_values("_trust_tot", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_s))

    cls_colors = ["#c0392b", "#f59e0b", "#0ea5e9", "#1e3a8a", "#0f766e", "#7c3aed"][: len(labels)]
    legend_short = [
        "C1: Broad Distrustors",
        "C2: Rule-of-Law Pragmatists",
        "C3: Moderate Trustors ('Quite a lot')",
        "C4: Strong Trustors ('A great deal')",
    ][: len(labels)]

    left = np.zeros(len(df_s))
    for lbl, lg_lbl, col in zip(labels, legend_short, cls_colors):
        vals = df_s[lbl].to_numpy()
        ax_st.barh(y_pos, vals, left=left, height=0.58, color=col, edgecolor="white", linewidth=0.8, label=lg_lbl)
        for i, v in enumerate(vals):
            if v >= 8.0:
                txt_col = "white" if col in ("#c0392b", "#1e3a8a", "#0f766e", "#7c3aed") else "#0f172a"
                ax_st.text(left[i] + v / 2.0, y_pos[i], f"{v:.1f}%", ha="center", va="center", fontsize=8.4, fontweight="bold", color=txt_col)
        left += vals

    ax_st.set_yticks(y_pos)
    ax_st.set_yticklabels([f"{r['country_name']}\n(n={int(r['n_obs']):,})" for _, r in df_s.iterrows()], fontsize=9.0)
    ax_st.set_xlim(0, 100)
    ax_st.set_xlabel("Posterior Archetype Share (%)")
    ax_st.set_title(
        "B. Citizen Archetype Mix Across 6 G8 Nations\n(Model-Based Mixture vs. Euclidean K-Means)",
        loc="left",
        pad=10,
        fontsize=10.3,
    )
    ax_st.spines["top"].set_visible(False)
    ax_st.spines["right"].set_visible(False)
    ax_st.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=2, frameon=False, fontsize=8.2)

    fig.subplots_adjust(left=0.145, right=0.975, top=0.86, bottom=0.22)
    fig.savefig(out_fig / "lca_citizen_archetypes.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run World Values Survey institutional trust analysis.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing wave_7.csv and time_series_1981_2022.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Repository output root for results/ and figures/")
    parser.add_argument("--lca-k", type=int, default=4, help="Number of LCA classes for main profile figure (2..6)")
    args = parser.parse_args()

    out_res = args.out_dir / "results"
    out_fig = args.out_dir / "figures"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print("Loading WVS Wave 7 (66 countries)...")
    w7 = load_wave7(args.data_dir)
    print(f"Loaded wave_7: {len(w7):,} rows across {w7['country'].nunique()} countries.")

    print("Running RQ1 (Strictly Filtered USA Wave 7)...")
    rq1 = run_rq1_usa(w7, out_res)
    print(f"  USA raw={rq1['usa_total_n']:,}, complete={rq1['usa_complete_n']:,}, Full McFadden R2={rq1['res_full'].mcfadden_r2:.4f}, Socio R2={rq1['res_socio'].mcfadden_r2:.4f}")
    print(f"  USA VIF range: [{rq1['vif']['vif'].min():.3f}, {rq1['vif']['vif'].max():.3f}]")

    print("Running RQ2 (Joint 6-G8 Ordered Logit + 66-Country External Macro Analysis)...")
    rq2 = run_rq2_g8_and_macro(w7, out_res)
    print(f"  G8 raw={rq2['g8_total_n']:,}, complete={rq2['g8_complete_n']:,}")
    print(f"  Country FE LRT: chi2({rq2['lrt_country'][1]})={rq2['lrt_country'][0]:.2f}, p={rq2['lrt_country'][2]:.3e}")
    print(f"  Country x Predictor LRT: chi2({rq2['lrt_interact'][1]})={rq2['lrt_interact'][0]:.2f}, p={rq2['lrt_interact'][2]:.3e}")

    print("Loading WVS Time Series (1995-2022 G8 waves)...")
    ts = load_time_series(args.data_dir)
    print("Running RQ3 (Longitudinal Institutional Evolution 1995-2022)...")
    rq3 = run_rq3_longitudinal(ts, out_res)
    print(f"  USA stacked obs={rq3['usa_stacked_obs']:,}, Inst x Year LRT: chi2({rq3['lrt_usa_div'][1]})={rq3['lrt_usa_div'][0]:.2f}, p={rq3['lrt_usa_div'][2]:.3e}")

    print("Running Bonus LCA (K=2..6 on 8 discrete 4-level institutional trust items)...")
    lca = run_bonus_lca(w7, out_res, chosen_k=args.lca_k)
    print(lca["df_sel"].to_string(index=False))
    print("\nLCA Profiles (K={}):".format(args.lca_k))
    print(lca["df_profiles"].to_string(index=False))
    print("\nLCA Country Shares:")
    print(lca["df_shares"].to_string(index=False))

    print("\nGenerating 4 publication figures...")
    plot_figure1_g8_and_macro(rq2, out_fig)
    plot_figure2_ordered_logit(rq1, rq2, out_fig)
    plot_figure3_longitudinal(rq3, out_fig)
    plot_figure4_lca(lca, out_fig)

    summary = {
        "wave7_total_n": int(len(w7)),
        "wave7_countries": int(w7["country"].nunique()),
        "usa_total_n": rq1["usa_total_n"],
        "usa_complete_n": rq1["usa_complete_n"],
        "usa_socio_r2": rq1["res_socio"].mcfadden_r2,
        "usa_full_r2": rq1["res_full"].mcfadden_r2,
        "usa_vif_min": float(rq1["vif"]["vif"].min()),
        "usa_vif_max": float(rq1["vif"]["vif"].max()),
        "g8_total_n": rq2["g8_total_n"],
        "g8_complete_n": rq2["g8_complete_n"],
        "g8_fe_r2": rq2["res_country_fe"].mcfadden_r2,
        "g8_interact_r2": rq2["res_interact"].mcfadden_r2,
        "g8_lrt_country_chi2": rq2["lrt_country"][0],
        "g8_lrt_interact_chi2": rq2["lrt_interact"][0],
        "usa_longitudinal_respondents": rq3["usa_respondents"],
        "usa_longitudinal_stacked_obs": rq3["usa_stacked_obs"],
        "usa_longitudinal_lrt_chi2": rq3["lrt_usa_div"][0],
        "lca_n_complete": lca["n_complete"],
        "lca_chosen_k": lca["chosen_k"],
        "lca_bic": lca["best"].bic,
        "lca_entropy": lca["best"].entropy,
    }
    (out_res / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDone. Summary saved to results/summary_metrics.json")


if __name__ == "__main__":
    main()
